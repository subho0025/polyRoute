"""
Loss functions for Imbalanced Polymer Classification in Pure TensorFlow.
Implements Class-Balanced Focal Loss with Label Smoothing.
Completely free of PyTorch.
"""

from typing import Optional, List
import tensorflow as tf
from tensorflow import keras

from .config import DEFAULT_CLASSES


class SparseCategoricalFocalCrossentropy(keras.losses.Loss):
    """
    Focal Crossentropy for integer-encoded classification labels.
    Downweights well-classified easy samples to focus learning on hard, minority classes.
    """

    def __init__(
        self,
        num_classes: int = len(DEFAULT_CLASSES),
        gamma: float = 2.0,
        alpha: float = 0.25,
        label_smoothing: float = 0.08,
        from_logits: bool = True,
        name: str = "sparse_categorical_focal_loss",
    ):
        super().__init__(name=name)
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.from_logits = from_logits
        self.focal = keras.losses.CategoricalFocalCrossentropy(
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
            from_logits=from_logits,
        )

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true_int = tf.cast(y_true, tf.int32)
        if len(y_true_int.shape) > 1 and y_true_int.shape[-1] == 1:
            y_true_int = tf.squeeze(y_true_int, axis=-1)
        y_true_one_hot = tf.one_hot(y_true_int, depth=self.num_classes)
        return self.focal(y_true_one_hot, y_pred)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
            "from_logits": self.from_logits,
        })
        return config


def get_loss_function(
    loss_type: str = "focal",
    num_classes: int = len(DEFAULT_CLASSES),
    label_smoothing: float = 0.08,
) -> keras.losses.Loss:
    """Factory function returning the specified loss function."""
    if loss_type.lower() == "focal":
        return SparseCategoricalFocalCrossentropy(
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.25,
            label_smoothing=label_smoothing,
            from_logits=True,
        )
    else:
        return keras.losses.SparseCategoricalCrossentropy(from_logits=True)
