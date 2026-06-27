# MudraVision

AI-powered Bharatanatyam mudra recognition using computer vision, deep learning, and MediaPipe.

---

## Overview

A computer vision project that recognizes Bharatanatyam mudras from images and webcam input using a deep learning classifier. The project is built to explore modern computer vision techniques including OpenCV, YOLO classification, MediaPipe hand landmark detection, and real-time inference.

The current implementation focuses on training and evaluating a baseline YOLO image classification model on a balanced Bharatanatyam mudra dataset. Future development will integrate MediaPipe hand landmarks to provide technique analysis and real-time feedback.

---

## Features

Current functionality:

* YOLO-based Bharatanatyam mudra classification
* Dataset preprocessing and organization
* Baseline model training
* Model validation
* Image prediction
* Evaluation across validation samples
* Incorrect prediction analysis

Planned functionality:

* Webcam inference
* MediaPipe hand landmark detection
* Real-time visual overlays
* Finger position analysis
* Technique feedback
* Voice feedback

---

## Project Structure

```text
mudra-vision/
│
├── dataset.yaml
├── requirements.txt
├── README.md
│
├── images/
│   ├── train/
│   └── val/
│
├── src/
│   ├── data/
│   │   └── organize_classification_dataset.py
│   │
│   ├── training/
│   │   ├── train_yolo_cls.py
│   │   ├── validate_yolo_cls.py
│   │   └── predict_yolo_cls.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_model.py
│   │   └── find_mistakes.py
│   │
│   └── opencv_basics.py
│
├── outputs/
│
├── runs/
│
└── models/
```

---

## Requirements

* Python 3.11+
* OpenCV
* Ultralytics
* PyTorch
* PyYAML
* Matplotlib

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset

The project uses the Bharatanatyam Mudra Dataset.

Current dataset statistics:

* 47 mudra classes
* 29,638 training images
* 7,662 validation images

Dataset layout:

```text
images/
├── train/
│   ├── Alapadmam/
│   ├── ...
│
└── val/
    ├── Alapadmam/
    ├── ...
```

---

## Training

Train the baseline YOLO classification model:

```bash
python src/training/train_yolo_cls.py
```

Training outputs are written to:

```text
runs/mudra_yolo_cls_baseline/
```

Generated files include:

```text
weights/
├── best.pt
└── last.pt

results.csv

confusion_matrix.png

args.yaml
```

`best.pt` contains the highest-performing model checkpoint.

---

## Validation

Evaluate the trained model:

```bash
python src/training/validate_yolo_cls.py
```

This computes classification metrics using the validation dataset.

---

## Image Prediction

Run inference on a single image:

```bash
python src/training/predict_yolo_cls.py
```

Output:

```text
Predicted mudra: Pathaka
Confidence: 98.4%
```

---

## Baseline Model Evaluation

Evaluate the trained model across validation samples:

```bash
python src/evaluation/evaluate_model.py
```

The evaluation script samples validation images from every mudra class and records:

* true label
* predicted label
* confidence score
* prediction correctness

Results are saved to:

```text
outputs/evaluation/sample_predictions.csv
```

---

## Analyze Incorrect Predictions

Inspect incorrect predictions:

```bash
python src/evaluation/find_mistakes.py
```

Example output:

```text
Image:
images/val/Pathaka/example.jpg

True:
Pathaka

Predicted:
Tripathaka

Confidence:
0.73
```

---

## OpenCV Fundamentals

Basic OpenCV image manipulation examples:

```bash
python src/opencv_basics.py
```

Demonstrates:

* image loading
* image dimensions
* pixel inspection
* drawing primitives
* image annotation
* image saving

---

## Development Workflow

1. Organize dataset
2. Train baseline classifier
3. Validate trained model
4. Predict on individual images
5. Evaluate model performance
6. Analyze failure cases
7. Integrate webcam inference
8. Add MediaPipe landmark detection
9. Implement real-time feedback

---


## License

This repository is intended for educational and research purposes.
