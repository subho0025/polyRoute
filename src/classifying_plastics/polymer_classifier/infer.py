"""
Inference Module for Polymer Classification in Pure TensorFlow.
Loads trained Keras models and provides single-image and batch predictions.
Completely free of PyTorch.
"""

import os
import json
from typing import Dict, Any, List, Optional, Union
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from .config import ClassifierConfigTF, DEFAULT_CLASSES, BASE_DIR
from .models import load_trained_model_tf


class PolymerPredictorTF:
    """Predicts polymer resin from input images using Pure TensorFlow."""

    def __init__(
        self,
        model: keras.Model,
        classes: List[str] = DEFAULT_CLASSES,
        class_mapping_path: Optional[str] = None,
    ):
        self.model = model
        self.classes = classes
        self.class_info = self._load_class_mapping(class_mapping_path)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Optional[str] = None,
        classes: List[str] = DEFAULT_CLASSES,
        class_mapping_path: Optional[str] = None,
    ) -> "PolymerPredictorTF":
        """Factory method loading predictor from saved Keras model."""
        model = load_trained_model_tf(model_path=checkpoint_path)
        return cls(model=model, classes=classes, class_mapping_path=class_mapping_path)

    def _load_class_mapping(self, mapping_path: Optional[str]) -> Dict[str, Any]:
        """Loads class metadata."""
        path = mapping_path or os.path.join(BASE_DIR, "class_mapping.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def preprocess_image(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """Preprocesses input image to (1, 224, 224, 3) float32 array."""
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        else:
            img = image_input.convert("RGB")

        img = img.resize((224, 224), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32)
        return arr[np.newaxis, ...]

    def predict(
        self,
        image_input: Union[str, Image.Image],
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """Runs inference on a single image."""
        tensor = self.preprocess_image(image_input)
        logits = self.model(tensor, training=False)
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        top_indices = np.argsort(probs)[::-1]
        top_class = self.classes[top_indices[0]]
        confidence = float(probs[top_indices[0]])

        all_probs = {cls_name: float(probs[i]) for i, cls_name in enumerate(self.classes)}

        top_classes = [
            {"class": self.classes[idx], "probability": float(probs[idx])}
            for idx in top_indices[:top_k]
        ]

        meta = self.class_info.get(top_class, {})

        return {
            "predicted_class": top_class,
            "confidence": confidence,
            "resin_code": meta.get("resin_code", "RIC"),
            "full_name": meta.get("full_name", top_class),
            "recyclability": meta.get("recyclability", "Recyclable"),
            "common_uses": meta.get("common_uses", "Packaging, containers"),
            "top_classes": top_classes,
            "all_probabilities": all_probs,
            "engine": "TensorFlow / Keras 3 (ConvNeXt-Tiny)",
        }

    def predict_batch(
        self,
        image_inputs: List[Union[str, Image.Image]],
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Runs batched inference across a list of images."""
        results = []
        for i in range(0, len(image_inputs), batch_size):
            chunk = image_inputs[i : i + batch_size]
            arrays = [self.preprocess_image(inp)[0] for inp in chunk]
            batch_tensor = np.stack(arrays, axis=0)

            logits = self.model(batch_tensor, training=False)
            batch_probs = tf.nn.softmax(logits, axis=-1).numpy()

            for j, probs in enumerate(batch_probs):
                top_idx = int(np.argmax(probs))
                top_class = self.classes[top_idx]
                conf = float(probs[top_idx])
                all_probs = {c: float(probs[k]) for k, c in enumerate(self.classes)}
                meta = self.class_info.get(top_class, {})

                results.append({
                    "predicted_class": top_class,
                    "confidence": conf,
                    "resin_code": meta.get("resin_code", "RIC"),
                    "full_name": meta.get("full_name", top_class),
                    "recyclability": meta.get("recyclability", "Recyclable"),
                    "all_probabilities": all_probs,
                })
        return results
