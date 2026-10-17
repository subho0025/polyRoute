# Classifying Plastics (Pure TensorFlow)

Automated plastic identification and polymer resin classification engine for circular recycling facilities.
Built natively in **TensorFlow 2.x** and **Keras 3** with zero PyTorch dependencies.

---

## Directory Overview

```text
classifying_plastics/
├── __init__.py                         # Python package interface
├── classify.py                         # Single-image and folder CLI inference tool
├── composition.py                      # Integration bridge for downstream Composition Estimator
├── evaluate.py                         # Evaluator script for held-out test split
├── generate_composition_inputs.py     # Generates predictions.json and confusion_matrix.json
├── requirements.txt                    # Pure TensorFlow dependencies
├── README.md                           # Documentation and integration guide
├── polymer_classification_pipeline_tf.ipynb # Self-contained Jupyter Notebook
├── tensorflow.ipynb                    # Alias notebook
├── data/
│   ├── predictions.json                # Ingestion payload for composition estimator
│   └── confusion_matrix.json           # 7x7 test set performance matrix
├── models/
│   ├── best_polymer_model_tf.keras     # Trained Keras model
│   ├── best_polymer_model_tf.weights.h5# Trained weights for fast in-place loading
│   ├── model_metadata.json             # Architecture & training metadata
│   ├── test_metrics_tf.json            # Precision, Recall, F1 scores per resin class
│   └── test_confusion_matrix_tf.png    # Confusion matrix heatmap
└── polymer_classifier/                 # Core modular Python library
    ├── __init__.py
    ├── config.py                       # Paths and dynamic discovery
    ├── dataset.py                      # Pure tf.data input pipelines
    ├── evaluator.py                    # Classification metrics & heatmaps
    ├── infer.py                        # PolymerPredictorTF inference class
    ├── losses.py                       # SparseCategoricalFocalCrossentropy
    ├── models.py                       # MobileNetV3-Large & ConvNeXt builders
    ├── pipeline.py                     # TwoStagePipelineTF decision logic
    └── trainer.py                      # Keras training loops and callbacks
```

---

## Repository Integration

### Uploading to Your Repository
Place or copy this `classifying_plastics` folder into your repository root or `src/` directory:
```bash
# Example: copy into your repository
cp -r classifying_plastics /path/to/your_repository/
```

### Installation
Activate your virtual environment and install requirements:
```bash
pip install -r classifying_plastics/requirements.txt
```

---

## Quick Usage

### 1. Classify Single Image (CLI)
```bash
cd classifying_plastics

# Classify and print JSON output:
python3 classify.py path/to/image.jpg --json

# Classify and save to a JSON file:
python3 classify.py path/to/image.jpg -o my_prediction.json

# Auto-run demo classification on available sample crop:
python3 classify.py
```

### 2. Process an Entire Folder of Images
```bash
cd classifying_plastics
python3 classify.py path/to/crops_directory/ -o data/predictions.json
```

### 3. Downstream Composition Estimator Check
```bash
cd classifying_plastics
python3 -c "import json; from composition import estimate_composition; print(estimate_composition(json.load(open('data/predictions.json')), total_kg=50)['kg'])"
```

### 4. Interactive Jupyter Notebook
Open `polymer_classification_pipeline_tf.ipynb`:
- Run **Section 10** to paste any image path and visualize predictions with horizontal probability bar charts.
- Run **Section 13** to automatically run schema and quality assertions.
