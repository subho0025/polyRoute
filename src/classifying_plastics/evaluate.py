#!/usr/bin/env python3
"""
Evaluation Script in Pure TensorFlow.
Evaluates a trained Keras checkpoint on the held-out test split
and outputs performance metrics and a confusion matrix heatmap.
"""

import os
import sys
import argparse

# Verify zero torch dependencies
assert "torch" not in sys.modules, "Fatal: torch detected in pure TensorFlow execution!"

from polymer_classifier.config import ClassifierConfigTF, BASE_DIR
from polymer_classifier.dataset import create_tf_datasets
from polymer_classifier.evaluator import PolymerEvaluatorTF


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pure TensorFlow Polymer Classifier on Test Set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained Keras model checkpoint (.keras or .weights.h5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports and confusion matrix heatmap",
    )
    args = parser.parse_args()

    cfg = ClassifierConfigTF()
    checkpoint_candidates = [
        args.checkpoint,
        cfg.model_path,
        cfg.weights_path,
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.keras"),
        os.path.join(BASE_DIR, "models", "best_polymer_model_tf.weights.h5"),
    ]
    checkpoint_path = next((p for p in checkpoint_candidates if p and os.path.exists(p)), None)

    if not checkpoint_path:
        print(f"[ERROR] Checkpoint file not found in {cfg.save_dir}")
        return

    output_dir = args.output_dir or cfg.save_dir
    print(f"Evaluating checkpoint: {checkpoint_path}")

    _, _, test_ds, _, classes = create_tf_datasets(cfg)

    if test_ds is None:
        print(f"[ERROR] Test split not found in {os.path.join(cfg.data_dir, 'test')}")
        return

    evaluator = PolymerEvaluatorTF.from_checkpoint(checkpoint_path, classes=classes)
    evaluator.evaluate_dataset(test_ds, output_dir=output_dir, save_chart=True)


if __name__ == "__main__":
    main()
