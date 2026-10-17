"""
Configuration module for Pure TensorFlow Polymer Classifier.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


# Dynamic discovery of dataset and models directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)

def find_data_dir() -> str:
    """Finds the dataset_crops directory."""
    candidates = [
        os.path.join(PROJECT_ROOT, "dataset_crops"),
        os.path.join(WORKSPACE_ROOT, "shivanshu", "dataset_crops"),
        os.path.join(WORKSPACE_ROOT, "dataset_crops"),
        os.path.join(os.getcwd(), "dataset_crops"),
        os.path.join(os.getcwd(), "shivanshu", "dataset_crops"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return c
    return os.path.join(PROJECT_ROOT, "dataset_crops")


def find_models_dir() -> str:
    """Finds the models directory."""
    candidates = [
        os.path.join(PROJECT_ROOT, "models"),
        os.path.join(WORKSPACE_ROOT, "models"),
        os.path.join(WORKSPACE_ROOT, "shivanshu", "models"),
    ]
    for c in candidates:
        if os.path.exists(c) and os.path.isdir(c):
            return c
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


BASE_DIR = PROJECT_ROOT
DATA_DIR = find_data_dir()
MODELS_DIR = find_models_dir()

# Internal 6 dataset classes
DEFAULT_CLASSES: List[str] = ["PE-HD", "PE-LD", "PET", "PP", "PS", "PVC"]

# Standard 7 polymer classes required for downstream recycling & composition estimation
ESTIMATOR_POLYMERS: List[str] = ["PET", "HDPE", "PVC", "LDPE", "PP", "PS", "Other"]

# Mapping from internal dataset folder names to standard 7 polymer classes
INTERNAL_TO_ESTIMATOR_MAP: Dict[str, str] = {
    "PE-HD": "HDPE",
    "PE-LD": "LDPE",
    "PET": "PET",
    "PP": "PP",
    "PS": "PS",
    "PVC": "PVC",
    "Other": "Other",
}


@dataclass
class ClassifierConfigTF:
    """Configuration dataclass for Pure TensorFlow training and inference."""

    # Model architecture (Optimized for < 20 GB training environments)
    model_name: str = "mobilenet_v3_large"  # mobilenet_v3_large (3.1M params), efficientnet_v2_b0 (6.1M), convnext_tiny
    input_shape: Tuple[int, int, int] = (224, 224, 3)
    num_classes: int = len(DEFAULT_CLASSES)
    dropout_rate: float = 0.2

    # Optimization
    epochs: int = 12
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.08
    loss_type: str = "focal"  # focal, categorical_crossentropy

    # Thresholds
    plastic_confidence_threshold: float = 0.60

    # Paths
    data_dir: str = field(default_factory=find_data_dir)
    save_dir: str = field(default_factory=find_models_dir)
    model_path: str = field(
        default_factory=lambda: os.path.join(find_models_dir(), "best_polymer_model_tf.keras")
    )
    weights_path: str = field(
        default_factory=lambda: os.path.join(find_models_dir(), "best_polymer_model_tf.weights.h5")
    )
    class_mapping_path: str = field(
        default_factory=lambda: os.path.join(PROJECT_ROOT, "class_mapping.json")
    )

    # Random seed
    seed: int = 42
