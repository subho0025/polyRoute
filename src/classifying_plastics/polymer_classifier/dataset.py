"""
Pure TensorFlow / tf.data Input Pipeline for Polymer Classification.
Completely free of PyTorch.
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from .config import ClassifierConfigTF, DEFAULT_CLASSES


def get_data_augmentation_pipeline() -> keras.Sequential:
    """Creates a pure TensorFlow Keras preprocessing augmentation pipeline."""
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.15),
            layers.RandomZoom(0.10),
            layers.RandomContrast(0.10),
        ],
        name="data_augmentation",
    )


def compute_class_weights(
    train_dir: str,
    classes: List[str] = DEFAULT_CLASSES,
    mode: str = "balanced",
) -> Dict[int, float]:
    """
    Computes class weights to counterbalance extreme real-world resin imbalance (e.g. 37:1).
    Modes:
      - 'balanced': n_samples / (n_classes * class_count)
      - 'sqrt': sqrt(max_count / class_count)
    """
    counts = []
    for cls_name in classes:
        cls_folder = os.path.join(train_dir, cls_name)
        if os.path.exists(cls_folder):
            counts.append(len([f for f in os.listdir(cls_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))
        else:
            counts.append(1)

    total_samples = sum(counts)
    num_classes = len(classes)
    weights = {}

    if mode == "sqrt":
        max_c = max(counts)
        for i, c in enumerate(counts):
            weights[i] = float(np.sqrt(max_c / max(1, c)))
    else:  # balanced
        for i, c in enumerate(counts):
            weights[i] = float(total_samples / (num_classes * max(1, c)))

    return weights


def create_tf_datasets(
    config: Optional[ClassifierConfigTF] = None,
    augment: bool = True,
) -> Tuple[tf.data.Dataset, Optional[tf.data.Dataset], Optional[tf.data.Dataset], List[int], List[str]]:
    """
    Builds optimized tf.data.Dataset pipelines for train, validation, and test splits.
    """
    cfg = config or ClassifierConfigTF()
    data_dir = cfg.data_dir
    image_size = cfg.input_shape[:2]
    batch_size = cfg.batch_size
    classes = DEFAULT_CLASSES

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    # Count samples per class in train
    class_counts = []
    for c in classes:
        p = os.path.join(train_dir, c)
        if os.path.exists(p):
            class_counts.append(len([f for f in os.listdir(p) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]))
        else:
            class_counts.append(0)

    # 1. Training dataset
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        class_names=classes,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=cfg.seed,
    )

    if augment:
        aug_layer = get_data_augmentation_pipeline()
        train_ds = train_ds.map(
            lambda x, y: (aug_layer(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # 2. Validation dataset
    val_ds = None
    if os.path.exists(val_dir):
        val_ds = keras.utils.image_dataset_from_directory(
            val_dir,
            labels="inferred",
            label_mode="int",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        ).prefetch(tf.data.AUTOTUNE)

    # 3. Test dataset
    test_ds = None
    if os.path.exists(test_dir):
        test_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            labels="inferred",
            label_mode="int",
            class_names=classes,
            image_size=image_size,
            batch_size=batch_size,
            shuffle=False,
        ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_counts, classes
