# MAI - IDL 2026 — Final Project Assignment

## Author

| Field | Detail |
|---|---|
| **Name** | Shreyas Hosadurga Sadananda |
| **Enrollment Number** | *(add your enrollment number here)* |
| **Repository** | https://github.com/Shreyasgowda6/idl26-finalassignment-shreyas_hosadurga_sadananda |

---

## Project Summary

This repository contains the completed final assignment for Introduction to Deep Learning (IDL) 2026. The task involved auditing and repairing a corrupted medical image classification pipeline, then extending it with an efficient model variant and transfer learning on a scarce dataset.

The codebase supports training and evaluating three CNN architectures — **ResNet18**, **VGG16**, and **AlexNet** — across four medical imaging datasets (`cells`, `chest`, `lesions`, `orgs`), plus transfer learning on a fifth small dataset (`organs`).

---

## Repository Structure

```
idl26-finalassignment-shreyas_hosadurga_sadananda/
│
├── Code/
│   ├── data.py            # Dataset loading and train/val/test splitting
│   ├── models.py          # AlexNet, VGG16, ResNet18 model definitions
│   ├── train.py           # Training entry point (reads config.json)
│   ├── fit.py             # Trainer class: training loop and evaluation
│   ├── evaluate.py        # Test set evaluation: accuracy, precision, recall, F1
│   ├── config.json        # Central config: dataset, model, hyperparameters
│   └── data/              # Dataset .pt files — download separately (link below)
│
├── AUDIT_LOG.md           # Bug audit: 14 bugs found and fixed across 4 files
├── REPORT.md              # Benchmark results and architectural recommendations
├── assignment_final.pdf   # Original assignment description
└── README.md              # This file
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/Shreyasgowda6/idl26-finalassignment-shreyas_hosadurga_sadananda.git
cd idl26-finalassignment-shreyas_hosadurga_sadananda
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision numpy scikit-learn
```

### 4. Download the datasets

Download from: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place all `.pt` files into `Code/data/`:

```
Code/data/
    cells.pt
    chest.pt
    lesions.pt
    orgs.pt
    organs.pt
```

---

## How to Run

### Train a model

Edit `Code/config.json` to set your dataset and model, then run from inside `Code/`:

```bash
cd Code
python3 train.py
```

Example `config.json`:

```json
{
    "DATA": "cells",
    "DATA_PATH": "data",
    "BATCH_SIZE": 32,
    "MODEL": "ResNet18",
    "CHANNELS": 3,
    "NUM_CLASSES": 8,
    "DROP_RATE": 0.5,
    "LEARNING_RATE": 0.001,
    "EPOCHS": 10
}
```

**Config reference per dataset:**

| Dataset | CHANNELS | NUM_CLASSES |
|---|---|---|
| cells | 3 | 8 |
| chest | 1 | 2 |
| lesions | 3 | 7 |
| orgs | 1 | 11 |
| organs | 1 | 11 |

Trained model weights are saved automatically to `Code/checkpoints/{DATA}_{MODEL}.pth`.

### Evaluate a trained model

```bash
cd Code
python3 evaluate.py
```

Reports accuracy, precision, recall, and macro F1 on the held-out test set.

---

## Results Summary

| Dataset | Best Model | Accuracy | Macro F1 | Min Target | Pass |
|---|---|---|---|---|---|
| cells | AlexNet | 94.50% | 0.9366 | 90% | ✅ |
| chest | ResNet18 | 91.51% | 0.9065 | 87% | ✅ |
| lesions | ResNet18 | 71.67% | 0.4466 | 67% | ✅ |
| orgs | ResNet18 | 92.83% | 0.9178 | 83% | ✅ |

Full results across all 12 model/dataset combinations: [REPORT.md](REPORT.md)

---

## What Was Fixed

14 bugs were identified and fixed across 4 files:

| File | Bugs Fixed |
|---|---|
| `data.py` | Wrong filename pattern, data leakage in train/val split |
| `models.py` | Broken activation function, VGGBlock channel propagation bug, AlexNet hardcoded channels/classes, wrong Linear input sizes, missing return statement in ResNet18 |
| `train.py` | Extreme dropout (0.99), dead kwarg argument, missing MPS device support for Apple Silicon |
| `fit.py` | Missing `zero_grad()`, wrong label shape `(N,1)` vs `(N,)`, variable `sum` shadowing Python built-in |

Full bug-by-bug audit with before/after code and explanations: [AUDIT_LOG.md](AUDIT_LOG.md)

---

## Original Assignment Information

Welcome to the official repository template for the **Introduction to Deep Learning (IDL) 2026 Final Assignment**.

### Overview

This repository contains the volatile, recovered remnants of a broken machine learning pipeline. Your mission is to audit the codebase, stabilize the system, optimize its computational footprint, and successfully deploy models across all target datasets.

- **Code:** All core source files can be found inside the `Code/` directory.
- **Instructions:** Background story and tasks are detailed in **`assignment_final.pdf`**.
- **Data:** Available for download here: https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

### Submission Guidelines

- **Platform:** Submit your final deliverables via the official **e-learning platform**.
- **Format:** Your submission must consist of a **direct link** to your created repository.
- **Branch:** Ensure all your final, production-ready code, your `AUDIT_LOG.md`, and your `REPORT.md` are completely merged into the **`main`** branch before the cutoff.
- **Deadline:** 09.07.2026, 23:59 (German Time). *Late submissions will not be processed.*
