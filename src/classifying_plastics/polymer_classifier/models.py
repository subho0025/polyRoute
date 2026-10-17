"""
Pure TensorFlow / Keras 3 Model Architectures for Polymer Classification.
Completely free of PyTorch.
"""

import os
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow import keras
from keras import layers

from .config import ClassifierConfigTF, DEFAULT_CLASSES


def build_classifier_tf(
    model_name: str = "mobilenet_v3_large",
    num_classes: int = len(DEFAULT_CLASSES),
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    dropout_rate: float = 0.2,
    pretrained: bool = True,
    trainable_backbone: bool = True,
) -> keras.Model:
    """
    Builds a vision classifier model in Pure TensorFlow / Keras 3.
    Default architecture is MobileNetV3-Large (~3.1M total parameters), specifically
    optimized for training under restricted memory constraints (< 20 GB total budget,
    < 1.5 GB peak training RAM).
    """
    inputs = keras.Input(shape=input_shape, name="input_image")
    weights = "imagenet" if pretrained else None

    # Select Backbone Architecture
    normalized_name = model_name.lower().replace("-", "_")
    if "mobilenet_v3_small" in normalized_name:
        backbone = keras.applications.MobileNetV3Small(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    elif "mobilenet_v3" in normalized_name or "mobilenet_v3_large" in normalized_name:
        backbone = keras.applications.MobileNetV3Large(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    elif "efficientnet" in normalized_name:
        backbone = keras.applications.EfficientNetV2B0(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    elif "mobilenet_v2" in normalized_name:
        backbone = keras.applications.MobileNetV2(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    elif "convnext" in normalized_name:
        backbone = keras.applications.ConvNeXtTiny(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    elif "resnet" in normalized_name:
        backbone = keras.applications.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )
    else:
        # Default to ultra-lightweight MobileNetV3Large
        backbone = keras.applications.MobileNetV3Large(
            include_top=False,
            weights=weights,
            input_tensor=inputs,
            pooling="avg",
        )

    backbone.trainable = trainable_backbone
    pooled = backbone.output

    # Projection Classification Head tailored for low parameter footprint and fast convergence
    if "convnext" in normalized_name or "resnet" in normalized_name:
        x = layers.LayerNormalization(name="classifier_layernorm")(pooled)
        x = layers.Dropout(dropout_rate, name="classifier_dropout1")(x)
        x = layers.Dense(384, activation="gelu", name="classifier_dense1")(x)
        x = layers.Dropout(dropout_rate * 0.67, name="classifier_dropout2")(x)
    else:
        # Ultra-efficient head for MobileNetV3 / EfficientNetV2 (< 150K head params)
        x = layers.Dense(128, activation="hard_swish", name="classifier_dense1")(pooled)
        x = layers.Dropout(dropout_rate, name="classifier_dropout")(x)

    logits = layers.Dense(num_classes, activation="linear", name="classifier_logits")(x)

    model = keras.Model(inputs=inputs, outputs=logits, name=f"polymer_classifier_{normalized_name}")
    return model


def load_trained_model_tf(
    model_path: Optional[str] = None,
    weights_path: Optional[str] = None,
    config: Optional[ClassifierConfigTF] = None,
) -> keras.Model:
    """
    Loads a trained Keras model or initializes the architecture and loads weights.
    100% Pure TensorFlow.
    """
    cfg = config or ClassifierConfigTF()
    target_model_path = model_path or cfg.model_path
    target_weights_path = weights_path or cfg.weights_path

    # Option 1: Load full saved .keras model
    if os.path.exists(target_model_path):
        try:
            model = keras.models.load_model(target_model_path, compile=False)
            return model
        except Exception:
            pass

    # Option 2: Build model and load .weights.h5
    model = build_classifier_tf(
        model_name=cfg.model_name,
        num_classes=cfg.num_classes,
        input_shape=cfg.input_shape,
        dropout_rate=cfg.dropout_rate,
        pretrained=False,
    )

    if os.path.exists(target_weights_path):
        model.load_weights(target_weights_path)
    elif os.path.exists(target_model_path):
        model.load_weights(target_model_path)
    else:
        raise FileNotFoundError(
            f"No trained checkpoint found at '{target_model_path}' or '{target_weights_path}'"
        )

    return model
