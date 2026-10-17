"""
Two-Stage Identification & Polymer Classification Pipeline in Pure TensorFlow.
Implements the exact user requirements:
Inputs: (photos, confidence limits/thresholds, batch metadata)
   |
   v
1. Plastic filter ........ drops non-plastic items; low-confidence items -> "manual check"
   |
   v
2. Polymer classifier .... PET / HDPE / PVC / LDPE / PP / PS / Other (+ confidence)

Completely free of PyTorch.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import tensorflow as tf

from .config import (
    BASE_DIR,
    DEFAULT_CLASSES,
    ESTIMATOR_POLYMERS,
    INTERNAL_TO_ESTIMATOR_MAP,
)
from .infer import PolymerPredictorTF


class TwoStagePipelineTF:
    """
    Two-Stage Waste Sorter:
    Stage 1: Plastic Filter (Drops non-plastic, routes ambiguous items to manual check)
    Stage 2: Polymer Classifier (Resin classification into 7 standard categories)
    """

    def __init__(
        self,
        predictor: PolymerPredictorTF,
        confidence_threshold: float = 0.60,
    ):
        self.predictor = predictor
        self.confidence_threshold = confidence_threshold

    def process_item(
        self,
        image_input: Union[str, Image.Image],
        plastic_probability: float = 1.0,
        item_id: Optional[str] = None,
        custom_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Executes the Two-Stage decision logic for a single optical item.
        """
        thresh = custom_threshold if custom_threshold is not None else self.confidence_threshold
        img_name = item_id or (os.path.basename(image_input) if isinstance(image_input, str) else "item_crop.jpg")

        # -----------------------------------------------------------------
        # STAGE 1: Plastic Filter
        # -----------------------------------------------------------------
        filter_confidence = max(plastic_probability, 1.0 - plastic_probability)

        # Condition 1A: Ambiguous / Low Confidence -> Manual Check
        if filter_confidence < thresh:
            return {
                "image": img_name,
                "material": "uncertain",
                "polymer": None,
                "confidence": round(float(filter_confidence), 4),
                "routing": "manual check",
                "stage1_status": "flagged_ambiguous",
                "stage2_status": "bypassed",
            }

        # Condition 1B: Definitively Non-Plastic (metal, cardboard, glass) -> Dropped
        if plastic_probability < 0.50:
            return {
                "image": img_name,
                "material": "non_plastic",
                "polymer": None,
                "confidence": round(float(1.0 - plastic_probability), 4),
                "routing": "drop / divert to alternative stream",
                "stage1_status": "dropped_non_plastic",
                "stage2_status": "bypassed",
            }

        # -----------------------------------------------------------------
        # STAGE 2: Polymer Classifier (for Verified Plastics)
        # -----------------------------------------------------------------
        pred_res = self.predictor.predict(image_input)
        raw_probs = pred_res["all_probabilities"]

        # Map internal 6-class probabilities to standard 7 classes
        mapped_probs = {p: 0.0 for p in ESTIMATOR_POLYMERS}
        for internal_cls, prob in raw_probs.items():
            est_name = INTERNAL_TO_ESTIMATOR_MAP.get(internal_cls, "Other")
            if est_name in mapped_probs:
                mapped_probs[est_name] += float(prob)

        # Normalize to ensure sum is exactly 1.0
        total = sum(mapped_probs.values())
        if total > 0:
            mapped_probs = {k: v / total for k, v in mapped_probs.items()}

        rounded_probs = {k: round(v, 4) for k, v in mapped_probs.items()}
        diff = round(1.0 - sum(rounded_probs.values()), 4)
        top_k = max(rounded_probs, key=rounded_probs.get)
        rounded_probs[top_k] = round(rounded_probs[top_k] + diff, 4)

        top_polymer = max(rounded_probs, key=rounded_probs.get)
        top_conf = rounded_probs[top_polymer]

        return {
            "image": img_name,
            "material": "plastic",
            "polymer": top_polymer,
            "confidence": top_conf,
            "probs": rounded_probs,
            "routing": f"optical jet sorter bin: {top_polymer}",
            "stage1_status": "verified_plastic",
            "stage2_status": "classified",
        }

    def process_batch(
        self,
        image_paths: List[str],
        output_json_path: Optional[str] = "data/predictions.json",
        default_plastic_prob: float = 1.0,
        batch_size: int = 32,
    ) -> List[Dict[str, Any]]:
        """Processes a batch of images and saves schema-compliant data/predictions.json using batched inference."""
        records = []
        filter_conf = max(default_plastic_prob, 1.0 - default_plastic_prob)

        if filter_conf < self.confidence_threshold:
            for p in image_paths:
                records.append({
                    "image": os.path.basename(p),
                    "material": "uncertain",
                    "polymer": None,
                    "confidence": round(float(filter_conf), 4),
                })
        elif default_plastic_prob < 0.50:
            for p in image_paths:
                records.append({
                    "image": os.path.basename(p),
                    "material": "non_plastic",
                    "polymer": None,
                    "confidence": round(float(1.0 - default_plastic_prob), 4),
                })
        else:
            # Verified plastic: batch inference through Stage 2
            for i in range(0, len(image_paths), batch_size):
                chunk_paths = image_paths[i : i + batch_size]
                batch_imgs = []
                for p in chunk_paths:
                    img = Image.open(p).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
                    batch_imgs.append(np.array(img, dtype=np.float32))

                batch_arr = np.stack(batch_imgs, axis=0)
                logits = self.predictor.model(batch_arr, training=False)
                batch_probs = tf.nn.softmax(logits, axis=-1).numpy()

                for p, probs in zip(chunk_paths, batch_probs):
                    mapped_probs = {p_cls: 0.0 for p_cls in ESTIMATOR_POLYMERS}
                    for internal_cls, prob_val in zip(self.predictor.classes, probs):
                        est_name = INTERNAL_TO_ESTIMATOR_MAP.get(internal_cls, "Other")
                        if est_name in mapped_probs:
                            mapped_probs[est_name] += float(prob_val)

                    total = sum(mapped_probs.values())
                    if total > 0:
                        mapped_probs = {k: v / total for k, v in mapped_probs.items()}

                    rounded_probs = {k: round(v, 4) for k, v in mapped_probs.items()}
                    diff = round(1.0 - sum(rounded_probs.values()), 4)
                    top_k = max(rounded_probs, key=rounded_probs.get)
                    rounded_probs[top_k] = round(rounded_probs[top_k] + diff, 4)

                    top_polymer = max(rounded_probs, key=rounded_probs.get)
                    top_conf = rounded_probs[top_polymer]

                    records.append({
                        "image": os.path.basename(p),
                        "material": "plastic",
                        "polymer": top_polymer,
                        "confidence": top_conf,
                        "probs": rounded_probs,
                    })

        if output_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            print(f"[INFO] Pipeline predictions saved to: {output_json_path} ({len(records)} records)")

        return records

    def export_confusion_matrix(
        self,
        test_ds: tf.data.Dataset,
        output_path: str = "data/confusion_matrix.json",
        source_tag: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes 7x7 confusion matrix across held-out test split in pure TensorFlow.
        Exports data/confusion_matrix.json.
        """
        all_targets = []
        all_preds = []

        for batch_images, batch_labels in test_ds:
            logits = self.predictor.model(batch_images, training=False)
            preds = np.argmax(logits.numpy(), axis=-1)

            all_targets.extend(batch_labels.numpy().tolist())
            all_preds.extend(preds.tolist())

        # Map to standard 7 classes
        classes = self.predictor.classes
        mapped_true = [INTERNAL_TO_ESTIMATOR_MAP[classes[t]] for t in all_targets]
        mapped_pred = [INTERNAL_TO_ESTIMATOR_MAP[classes[p]] for p in all_preds]

        cm = confusion_matrix(mapped_true, mapped_pred, labels=ESTIMATOR_POLYMERS)

        date_str = datetime.now().strftime("%d %b %Y")
        total_imgs = len(all_targets)
        source_text = source_tag or f"Pure TensorFlow Polymer Classifier test set, {total_imgs} images, {date_str}"

        payload = {
            "source": source_text,
            "labels": ESTIMATOR_POLYMERS,
            "matrix": cm.tolist(),
        }

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[INFO] 7x7 confusion matrix saved to: {output_path}")
        print(f"   Shape: {cm.shape} with labels: {ESTIMATOR_POLYMERS}")
        return payload
