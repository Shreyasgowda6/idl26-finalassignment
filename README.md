# MAI - IDL 2026 Final Project Assignment

## Team Members

| Name | Enrollment Number |
|---|---|
| Shreyas Hosadurga Sadananda | 10013494 |
| Suhaim Manna | 10012548 |

Repositories:

- https://github.com/Shreyasgowda6/idl26-finalassignment
- https://github.com/suhaimz11/idl26-finalassignment

---

## Overview

This repository contains the completed final assignment for Introduction to Deep Learning (IDL) 2026. The project restores a broken medical image classification pipeline and extends it with full benchmarking, green-model profiling, and transfer learning for a scarce-data setting.

The main benchmark evaluates **AlexNet**, **VGG16**, and **ResNet18** on four medical imaging datasets:

- `cells`
- `chest`
- `lesions`
- `orgs`

The project also includes:

- an efficient MiniResNet model and full green-efficiency profiling for the Green Initiative analysis
- transfer learning from orgs to the smaller organs dataset
- CSV logging for benchmark, evaluation, green-profile, and transfer-learning results

---

## Repository Structure

```text
idl26-finalassignment/
|
|-- Code/
|   |-- data.py                 # Dataset loading and train/validation/test splitting
|   |-- fit.py                  # Trainer class and training loop
|   |-- models.py               # AlexNet, VGG16, ResNet18, MiniResNet
|   |-- train.py                # Single-run training entry point
|   |-- train_all.py            # Full benchmark runner
|   |-- evaluate.py             # Checkpoint evaluation and CSV logging
|   |-- green_profile.py        # Green-model profiling for Part 2
|   |-- transfer_learning.py    # Scarce-data transfer learning for Part 3
|   |-- config.json             # Single-run configuration
|   |-- config_all.json         # Full benchmark configuration
|
|-- data/                       # Dataset .pt files, not committed
|-- outputs/                    # Generated CSV result files
|-- AUDIT_LOG.md                # Bug audit and implemented fixes
|-- REPORT.md                   # Benchmark, green-model, and transfer-learning report
|-- assignment_final.pdf        # Assignment description
|-- README.md                   # Project documentation
```

---

## Setup

Create and activate a Python environment:

```bash
python -m venv venv
```
```

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn
```

For CUDA GPU support, install a CUDA-enabled PyTorch build that matches the local GPU, driver, and Python version.

---

## Data

Download the assignment datasets from:

https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place the .pt files in the root-level data/ folder:

```text
data/
|-- cells.pt
|-- chest.pt
|-- lesions.pt
|-- orgs.pt
|-- organs.pt
```

Dataset configuration used by the pipeline:

| Dataset | Channels | Classes |
|---|---:|---:|
| cells | 3 | 8 |
| chest | 1 | 2 |
| lesions | 3 | 7 |
| orgs | 1 | 11 |
| organs | 1 | 11 |

---

## Running the Code

Run commands from inside the Code/ directory:

```bash
cd Code
```

### Train One Model

Edit config.json, then run:

```bash
python train.py
```

Checkpoints are saved to:

```text
Code/checkpoints/{DATA}_{MODEL}.pth
```

### Evaluate One Model

```bash
python evaluate.py
```

### Evaluate All Checkpoints

```bash
python evaluate.py --all
```

This writes:

```text
outputs/evaluation_results.csv
```

### Run the Full Benchmark

```bash
python train_all.py
```

This uses config_all.json and writes:

```text
outputs/benchmark_results.csv
```

### Run Green Model Profiling

```bash
python green_profile.py
```

This profiles AlexNet, VGG16, ResNet18, and MiniResNet across the four main datasets and writes:

```text
outputs/green_profile_results.csv
```

### Run Transfer Learning

```bash
python transfer_learning.py
```

This compares training from scratch against transfer learning from the larger orgs checkpoint to the smaller organs dataset and writes:

```text
outputs/transfer_learning_results.csv
```

---

## Results Summary

| Dataset | Best Model | Accuracy | Macro F1 | Target | Pass |
|---|---|---:|---:|---:|---|
| cells | ResNet18 | 97.16% | 0.9719 | 90% | Yes |
| chest | ResNet18 | 90.06% | 0.8890 | 87% | Yes |
| lesions | ResNet18 | 76.61% | 0.4834 | 67% | Yes |
| orgs | VGG16 | 90.06% | 0.8857 | 83% | Yes |

All 12 main benchmark runs passed the required target accuracies. The detailed benchmark discussion, dataset-wise green profiling, and transfer-learning analysis are available in [REPORT.md](REPORT.md).

---

## Codebase Fix Summary

| File | Main Work Completed |
|---|---|
| data.py | Corrected dataset file loading and fixed the train/validation split so validation samples are not included in training. |
| models.py | Fixed model architecture issues in VGGBlock, AlexNet, VGG16, and ResNet18 so all models work across the required datasets. Added MiniResNet for the Green Initiative analysis. |
| train.py | Rebuilt the training entry point around config.json, proper device selection, configurable dropout, and checkpoint saving. |
| fit.py | Fixed the training loop by resetting gradients each batch and using correctly shaped labels for CrossEntropyLoss. |

Detailed bug descriptions, root causes, and implemented corrections are documented in [AUDIT_LOG.md](AUDIT_LOG.md).

---

## Main Deliverables

| File | Purpose |
|---|---|
| AUDIT_LOG.md | Documents discovered bugs, root causes, fixes, and related commits |
| REPORT.md | Contains benchmark results, recommendations, Green Initiative analysis, and transfer-learning analysis |
| Code/ | Contains the corrected and extended training/evaluation pipeline |
| outputs/ | Contains generated CSV result files used for the report |

---

