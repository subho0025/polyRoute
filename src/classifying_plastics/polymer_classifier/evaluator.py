"""
Evaluation Module for Pure TensorFlow Polymer Classifier.
Calculates Accuracy, Macro F1, Weighted F1, and produces Confusion Matrix Heatmaps.
Completely free of PyTorch.
"""

import os
import json
from typing import Dict, Any, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from .config import ClassifierConfigTF, DEFAULT_CLASSES, BASE_DIR
from .models import load_trained_model_tf


class PolymerEvaluatorTF:
    """Evaluates Pure TensorFlow polymer classification models."""

    def __init__(
        self,
        model: keras.Model,
        classes: List[str] = DEFAULT_CLASSES,
        config: Optional[ClassifierConfigTF] = None,
    ):
        self.model = model
        self.classes = classes
        self.config = config or ClassifierConfigTF()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[str] = None,
        classes: List[str] = DEFAULT_CLASSES,
    ) -> "PolymerEvaluatorTF":
        """Factory method loading evaluator from saved Keras model."""
        model = load_trained_model_tf(model_path=checkpoint_path)
        return cls(model=model, classes=classes)

    def evaluate_dataset(
        self,
        test_ds: tf.data.Dataset,
        output_dir: Optional[str] = None,
        save_chart: bool = True,
    ) -> Dict[str, Any]:
        """
        Runs comprehensive evaluation over a tf.data.Dataset.
        Computes Accuracy, Macro F1, Per-Class Metrics, and Confusion Matrix.
        """
        all_targets = []
        all_preds = []
        all_probs = []

        print("[INFO] Running evaluation across test split...")
        for batch_images, batch_labels in test_ds:
            logits = self.model(batch_images, training=False)
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            preds = np.argmax(probs, axis=-1)

            all_preds.extend(preds.tolist())
            all_targets.extend(batch_labels.numpy().tolist())
            all_probs.extend(probs.tolist())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        # Calculate metrics
        acc = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        report_dict = classification_report(
            y_true,
            y_pred,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, y_pred, labels=range(len(self.classes)))

        metrics_payload = {
            "test_accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "total_samples": len(y_true),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes,
        }

        # Save artifacts
        target_dir = output_dir or self.config.save_dir
        os.makedirs(target_dir, exist_ok=True)

        json_path = os.path.join(target_dir, "test_metrics_tf.json")
        with open(json_path, "w") as f:
            json.dump(metrics_payload, f, indent=2)

        if save_chart:
            chart_path = os.path.join(target_dir, "test_confusion_matrix_tf.png")
            plt.figure(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.classes,
                yticklabels=self.classes,
            )
            plt.title(f"Polymer Classification Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro F1: {macro_f1*100:.2f}%")
            plt.xlabel("Predicted Polymer Class")
            plt.ylabel("Ground Truth Class")
            plt.tight_layout()
            plt.savefig(chart_path, dpi=200)
            plt.close()

        print(f"\n=======================================================")
        print(f"Test Benchmark Results:")
        print(f"   Accuracy    : {acc * 100:.2f}%")
        print(f"   Macro F1    : {macro_f1 * 100:.2f}%")
        print(f"   Weighted F1 : {weighted_f1 * 100:.2f}%")
        print(f"   Total Crops : {len(y_true)}")
        print(f"=======================================================\n")

        return metrics_payload
