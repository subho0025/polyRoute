"""
Trainer Module for Polymer Classification in Pure TensorFlow / Keras.
Completely free of PyTorch.
"""

import os
import json
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow import keras

from .config import ClassifierConfigTF
from .losses import get_loss_function


class PolymerTrainerTF:
    """
    Manages model compilation, training loops, callbacks, and checkpointing in TensorFlow.
    """

    def __init__(
        self,
        model: keras.Model,
        config: ClassifierConfigTF,
        train_ds: tf.data.Dataset,
        val_ds: Optional[tf.data.Dataset] = None,
        class_weights: Optional[Dict[int, float]] = None,
    ):
        self.model = model
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.class_weights = class_weights

        os.makedirs(self.config.save_dir, exist_ok=True)

    def compile_model(self):
        """Compiles model with AdamW optimizer, cosine decay, and focal loss."""
        # Cosine decay schedule
        lr_schedule = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.config.learning_rate,
            decay_steps=self.config.epochs * 500,
            alpha=0.01,
        )

        optimizer = keras.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=self.config.weight_decay,
        )

        loss_fn = get_loss_function(
            loss_type=self.config.loss_type,
            num_classes=self.config.num_classes,
            label_smoothing=self.config.label_smoothing,
        )

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    def get_callbacks(self) -> list:
        """Configures training callbacks."""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy" if self.val_ds is not None else "accuracy",
                patience=5,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=self.config.weights_path,
                monitor="val_accuracy" if self.val_ds is not None else "accuracy",
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            ),
        ]

        # CSV training logger
        log_path = os.path.join(self.config.save_dir, "training_history_tf.csv")
        callbacks.append(keras.callbacks.CSVLogger(log_path))

        return callbacks

    def train(self) -> Dict[str, Any]:
        """Runs the training process."""
        print(f"\n[INFO] Compiling TensorFlow model with {self.config.loss_type} loss...")
        self.compile_model()

        callbacks = self.get_callbacks()

        print(f"[INFO] Starting training for {self.config.epochs} epochs...")
        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=self.config.epochs,
            class_weight=self.class_weights,
            callbacks=callbacks,
            verbose=1,
        )

        # Save full .keras model
        self.model.save(self.config.model_path)
        print(f"[SUCCESS] Saved best full Keras model to: {self.config.model_path}")

        # Save history as JSON
        hist_dict = history.history
        # Convert any float32 to float for json serialization
        serializable_hist = {k: [float(v) for v in vals] for k, vals in hist_dict.items()}
        hist_json_path = os.path.join(self.config.save_dir, "training_history_tf.json")
        with open(hist_json_path, "w") as f:
            json.dump(serializable_hist, f, indent=2)

        return serializable_hist
