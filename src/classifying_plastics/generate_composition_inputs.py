#!/usr/bin/env python3
"""
CLI Tool: Generates Two-Stage Pipeline Inputs & Performance Matrix in Pure TensorFlow.
1. Predictions JSON (image, material, polymer, confidence, probs)
2. Confusion Matrix JSON (data/confusion_matrix.json)

Usage:
    # Generate both on held-out test split
    python generate_composition_inputs.py --all

    # Generate predictions for a specific image folder
    python generate_composition_inputs.py --images dataset_crops/test/PET --output data/predictions.json

    # Generate only the confusion matrix
    python generate_composition_inputs.py --confusion-matrix
"""

import os
import sys
import glob
import json
import argparse

# Verify zero torch dependencies
assert "torch" not in sys.modules, "Fatal: torch detected in pure TensorFlow execution!"

from polymer_classifier.config import ClassifierConfigTF, BASE_DIR
from polymer_classifier.infer import PolymerPredictorTF
from polymer_classifier.dataset import create_tf_datasets
from polymer_classifier.pipeline import TwoStagePipelineTF


def get_active_checkpoint():
    cfg = ClassifierConfigTF()
    candidates = [
        cfg.model_path,
        cfg.weights_path,
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.keras"),
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.weights.h5"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("No trained Keras model checkpoint found in models/.")


def main():
    parser = argparse.ArgumentParser(description="Generate Two-Stage Pipeline Inputs in Pure TensorFlow")
    parser.add_argument("--all", action="store_true", help="Generate both predictions and confusion matrix")
    parser.add_argument("--images", type=str, help="Directory or glob pattern of images to predict on")
    parser.add_argument("--output", type=str, default="data/predictions.json", help="Predictions JSON output path")
    parser.add_argument("--confusion-matrix", action="store_true", help="Generate data/confusion_matrix.json")
    parser.add_argument("--cm-output", type=str, default="data/confusion_matrix.json", help="Confusion matrix output path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Custom Keras checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.60, help="Confidence threshold for plastic filter manual check")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or get_active_checkpoint()
    print(f"Using checkpoint: {checkpoint_path}")

    predictor = PolymerPredictorTF.from_checkpoint(checkpoint_path)
    pipeline = TwoStagePipelineTF(predictor=predictor, confidence_threshold=args.threshold)

    # 1. Generate Confusion Matrix (7x7)
    if args.all or args.confusion_matrix:
        print("\n[1/2] Computing 7-Class Confusion Matrix...")
        cfg = ClassifierConfigTF()
        _, _, test_ds, _, _ = create_tf_datasets(cfg)
        if test_ds is not None:
            cm_payload = pipeline.export_confusion_matrix(test_ds, output_path=args.cm_output)
            print(f"   Confusion Matrix Source: {cm_payload['source']}")
            print(f"   Labels ({len(cm_payload['labels'])}): {cm_payload['labels']}")
        else:
            print("   [WARNING] Test dataset unavailable for confusion matrix.")

    # 2. Generate Predictions
    if args.all or args.images or not args.confusion_matrix:
        print("\n[2/2] Generating Predictions via Two-Stage Pipeline...")

        if args.images:
            if os.path.isdir(args.images):
                img_list = sorted(glob.glob(os.path.join(args.images, "**", "*.jpg"), recursive=True))
                if not img_list:
                    img_list = sorted(glob.glob(os.path.join(args.images, "*.png")))
            else:
                img_list = sorted(glob.glob(args.images))
        else:
            cfg = ClassifierConfigTF()
            img_list = sorted(glob.glob(os.path.join(cfg.data_dir, "test", "*", "*.jpg")))

        if not img_list:
            print("[ERROR] No images found to predict on.")
            return

        print(f"Processing {len(img_list)} images through Two-Stage Pipeline...")
        records = pipeline.process_batch(img_list, output_json_path=args.output)

        print("\nSample Output Records (first 3):")
        print(json.dumps(records[:3], indent=2))

    print("\n[INFO] Pipeline execution complete.")


if __name__ == "__main__":
    main()
