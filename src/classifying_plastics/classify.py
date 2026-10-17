#!/usr/bin/env python3
"""
CLI Tool for Plastic Polymer Classification in Pure TensorFlow.
Analyzes an image crop or a folder of crops, predicts polymer resin type,
and outputs/saves the exact JSON format required by the Composition Estimator.

Usage:
    # 1. Output clean JSON to terminal for a single image:
    python3 classify.py <path_to_image.jpg> --json

    # 2. Save JSON file for a single image:
    python3 classify.py <path_to_image.jpg> -o output.json

    # 3. Process an entire folder of images and save predictions.json:
    python3 classify.py <path_to_folder> -o predictions.json

    # 4. Human-readable summary report:
    python3 classify.py <path_to_image.jpg>
"""

import os
import sys

# Silence TensorFlow C++ allocator and hardware diagnostics
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse
import glob
import json
import tensorflow as tf
tf.get_logger().setLevel("ERROR")

# Ensure package is on PYTHONPATH
_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_PATHS = [
    _DIR,
    os.path.join(_DIR, "tensorflow_solution"),
    os.path.join(os.path.dirname(_DIR), "tensorflow_solution"),
    os.path.join(_DIR, "polymer_classifier"),
]
for p in _CANDIDATE_PATHS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Verify zero torch dependencies
assert "torch" not in sys.modules, "Fatal: torch detected in pure TensorFlow execution!"

from polymer_classifier.config import ClassifierConfigTF, BASE_DIR, INTERNAL_TO_ESTIMATOR_MAP, ESTIMATOR_POLYMERS
from polymer_classifier.infer import PolymerPredictorTF


def format_record(image_path: str, result: dict) -> dict:
    """Formats raw prediction into the Composition Estimator JSON schema."""
    top_class = result["predicted_class"]
    conf = result["confidence"]
    all_probs = result["all_probabilities"]

    mapped_probs = {p: 0.0 for p in ESTIMATOR_POLYMERS}
    for k, v in all_probs.items():
        est_name = INTERNAL_TO_ESTIMATOR_MAP.get(k, "Other")
        if est_name in mapped_probs:
            mapped_probs[est_name] += float(v)

    total = sum(mapped_probs.values())
    if total > 0:
        mapped_probs = {k: round(v / total, 4) for k, v in mapped_probs.items()}
        diff = round(1.0 - sum(mapped_probs.values()), 4)
        top_k = max(mapped_probs, key=mapped_probs.get)
        mapped_probs[top_k] = round(mapped_probs[top_k] + diff, 4)

    top_est = INTERNAL_TO_ESTIMATOR_MAP.get(top_class, "Other")
    return {
        "image": os.path.basename(image_path),
        "material": "plastic",
        "polymer": top_est,
        "confidence": round(float(conf), 4),
        "probs": mapped_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify Plastic Polymer Resin using Pure TensorFlow")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to an input image file or a directory containing images. Auto-selects a sample if omitted.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Save output JSON record(s) to the specified file path (e.g. -o output.json)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to Keras model checkpoint (.keras or .weights.h5). Defaults to best trained model.",
    )
    parser.add_argument("--top", type=int, default=3, help="Number of top classes to display in summary report")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON format for composition estimator to stdout",
    )
    args = parser.parse_args()

    # 1. Resolve image path or auto-pick sample test crop
    if not args.image_path:
        sample_candidates = (
            glob.glob("dataset_crops/test/*/*.jpg")
            + glob.glob("../dataset_crops/test/*/*.jpg")
            + glob.glob(os.path.join(BASE_DIR, "dataset_crops", "test", "*", "*.jpg"))
        )
        if sample_candidates:
            args.image_path = sample_candidates[0]
            if not args.json and not args.output:
                print(f"[INFO] No image argument provided. Auto-classifying sample crop:\n       {args.image_path}")
        else:
            print("[ERROR] Please specify an image or directory: python3 classify.py <path_to_image_or_dir>")
            sys.exit(1)

    if not os.path.exists(args.image_path):
        print(f"[ERROR] Path '{args.image_path}' not found.")
        sys.exit(1)

    # 2. Find trained checkpoint
    cfg = ClassifierConfigTF()
    checkpoint_candidates = [
        args.checkpoint,
        os.path.join(_DIR, "models", "best_polymer_model_tf.keras"),
        os.path.join(_DIR, "models", "best_polymer_model_tf.weights.h5"),
        cfg.model_path,
        cfg.weights_path,
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.keras"),
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.weights.h5"),
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p and os.path.exists(p)), None)

    if not checkpoint_path:
        print(f"[ERROR] No trained Keras checkpoint found.")
        sys.exit(1)

    predictor = PolymerPredictorTF.from_checkpoint(checkpoint_path)

    # 3. Handle Directory of Images
    if os.path.isdir(args.image_path):
        image_files = sorted(
            glob.glob(os.path.join(args.image_path, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(args.image_path, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(args.image_path, "**", "*.jpeg"), recursive=True)
        )
        if not image_files:
            print(f"[ERROR] No image files (.jpg, .png) found in directory '{args.image_path}'")
            sys.exit(1)

        if not args.json and not args.output:
            print(f"[INFO] Found {len(image_files)} images in '{args.image_path}'. Processing...")

        records = []
        for img_p in image_files:
            res = predictor.predict(img_p)
            records.append(format_record(img_p, res))

        if args.output:
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"[SUCCESS] Processed {len(records)} images. Saved predictions to: {args.output}")
        else:
            print(json.dumps(records, indent=2))
        return

    # 4. Handle Single Image
    result = predictor.predict(args.image_path, top_k=args.top)
    json_record = format_record(args.image_path, result)

    # Save to file if -o/--output specified
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump([json_record], f, indent=2)
        print(f"[SUCCESS] Saved JSON record to: {args.output}")
        return

    # Print JSON to terminal if --json specified
    if args.json:
        print(json.dumps(json_record, indent=2))
        return

    # Otherwise print human-readable summary report
    top_class = result["predicted_class"]
    conf = result["confidence"]
    full_name = result["full_name"]

    print("\n" + "=" * 70)
    print("        POLYMER RESIN CLASSIFICATION REPORT (TENSORFLOW)")
    print("=" * 70)
    print(f"Image Path        : {args.image_path}")
    print(f"Detected Polymer  : {top_class} ({full_name})")
    print(f"Confidence        : {conf * 100:.2f}%")
    print(f"Execution Engine  : {result.get('engine', 'TensorFlow / Keras 3')}")
    print(f"Common Uses       : {result['common_uses']}")
    print(f"Recyclability     : {result['recyclability']}")

    # Probability Distribution Table
    print("\nProbability Distribution Across Resin Classes:")
    print("-" * 70)
    all_probs = result["all_probabilities"]
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    for cls_name, prob in sorted_probs:
        print(f"  {cls_name:<8} : {prob * 100:>6.2f}%")

    print("=" * 70)
    print("[TIP] To save output as a JSON file, run:")
    print(f"      python3 classify.py {args.image_path} -o my_prediction.json\n")


if __name__ == "__main__":
    main()
