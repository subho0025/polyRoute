"""
Pure TensorFlow Polymer Classifier & Two-Stage Sorting Pipeline.
Zero PyTorch dependencies.
"""

from .config import (
    ClassifierConfigTF,
    DEFAULT_CLASSES,
    ESTIMATOR_POLYMERS,
    INTERNAL_TO_ESTIMATOR_MAP,
    BASE_DIR,
    DATA_DIR,
    MODELS_DIR,
)
from .models import build_classifier_tf, load_trained_model_tf
from .dataset import create_tf_datasets, compute_class_weights
from .losses import get_loss_function, SparseCategoricalFocalCrossentropy
from .trainer import PolymerTrainerTF
from .evaluator import PolymerEvaluatorTF
from .infer import PolymerPredictorTF
from .pipeline import TwoStagePipelineTF

__all__ = [
    "ClassifierConfigTF",
    "DEFAULT_CLASSES",
    "ESTIMATOR_POLYMERS",
    "INTERNAL_TO_ESTIMATOR_MAP",
    "BASE_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "build_classifier_tf",
    "load_trained_model_tf",
    "create_tf_datasets",
    "compute_class_weights",
    "get_loss_function",
    "SparseCategoricalFocalCrossentropy",
    "PolymerTrainerTF",
    "PolymerEvaluatorTF",
    "PolymerPredictorTF",
    "TwoStagePipelineTF",
]
