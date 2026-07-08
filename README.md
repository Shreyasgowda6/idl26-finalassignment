# MAI - IDL 2026 Final Project Assignment

## Team Members

| Name | Enrollment Number |
|---|---|
| Shreyas Hosadurga Sadananda | 10013494 |
| Suhaim Manna |  |

Repository:  https://github.com/Shreyasgowda6/idl26-finalassignment | https://github.com/suhaimz11/idl26-finalassignment

---

## Project Summary

This repository contains the completed final assignment for Introduction to Deep Learning (IDL) 2026. The project repairs and extends a recovered medical image classification pipeline. The final code supports training, evaluation, benchmarking, green model profiling, and transfer learning.

The main benchmark evaluates **AlexNet**, **VGG16**, and **ResNet18** on four medical imaging datasets:

- `cells`
- `chest`
- `lesions`
- `orgs`

The project also includes:

- an efficient **MiniResNet** model for the Green Initiative analysis
- transfer learning from `orgs` to the scarce `organs` dataset
- CSV logging for benchmark, evaluation, green-profile, and transfer-learning results

---

## Repository Structure

```text
idl26-finalassignment/
|
|-- Code/
|   |-- data.py                 # Dataset loading and train/val/test splitting
|   |-- fit.py                  # Trainer class and training loop
|   |-- models.py               # AlexNet, VGG16, ResNet18, MiniResNet
|   |-- train.py                # Single-run training entry point
|   |-- train_all.py            # Runs all dataset/model benchmark configs
|   |-- evaluate.py             # Evaluates checkpoints and saves CSV metrics
|   |-- green_profile.py        # Part 2 green-model profiling
|   |-- transfer_learning.py    # Part 3 scarce-data transfer learning
|   |-- config.json             # Single-run config
|   |-- config_all.json         # Full benchmark config
|
|-- data/                       # Dataset .pt files, not committed
|-- outputs/                    # Generated CSV result files, not committed
|-- AUDIT_LOG.md                # Bug audit and fixes
|-- REPORT.md                   # Benchmark, green-model, and transfer-learning report
|-- assignment_final.pdf        # Original assignment description
|-- README.md                   # This file
```

---

## Setup

Create and activate a Python environment:

```bash
python -m venv venv
```

On Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install torch torchvision numpy scikit-learn
```

For CUDA GPU support, install a CUDA-enabled PyTorch build appropriate for the local GPU and Python version.

---

## Data

Download the assignment datasets from:

https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

Place the `.pt` files in the root-level `data/` folder:

```text
data/
|-- cells.pt
|-- chest.pt
|-- lesions.pt
|-- orgs.pt
|-- organs.pt
```

The expected dataset metadata is:

| Dataset | Channels | Classes |
|---|---:|---:|
| cells | 3 | 8 |
| chest | 1 | 2 |
| lesions | 3 | 7 |
| orgs | 1 | 11 |
| organs | 1 | 11 |

---

## Running the Code

Run commands from inside the `Code/` directory:

```bash
cd Code
```

### Train One Model

Edit `config.json`, then run:

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

This uses `config_all.json` and writes:

```text
outputs/benchmark_results.csv
```

### Run Green Model Profiling

```bash
python green_profile.py
```

This compares ResNet18 and MiniResNet across all four main datasets and writes:

```text
outputs/green_profile_results.csv
```

### Run Transfer Learning

```bash
python transfer_learning.py
```

This compares scratch training against transfer learning from the larger `orgs` checkpoint to the smaller `organs` dataset and writes:

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

All 12 main benchmark runs passed the required target accuracies. Full benchmark details, green-model analysis, and transfer-learning analysis are available in [REPORT.md](REPORT.md).

---

## Main Deliverables

| File | Purpose |
|---|---|
| `AUDIT_LOG.md` | Documents discovered bugs, root causes, fixes, and commit hashes |
| `REPORT.md` | Contains benchmark results, recommendations, Green Initiative analysis, and transfer-learning analysis |
| `Code/` | Contains the corrected and extended training/evaluation pipeline |

---

## Original Assignment Information

Welcome to the official repository template for the **Introduction to Deep Learning (IDL) 2026 Final Assignment**.

### Overview

This repository contains the volatile, recovered remnants of a broken machine learning pipeline. The assignment requires auditing the codebase, stabilizing the system, optimizing its computational footprint, and deploying models across all target datasets.

- **Code:** Core source files are inside `Code/`.
- **Instructions:** Full task details are in `assignment_final.pdf`.
- **Data:** Available from https://cloud.fiw.fhws.de/s/LpYa2dCW85kwdNn

### Submission Guidelines

- **Platform:** Submit final deliverables through the official e-learning platform.
- **Format:** Submission must include a direct link to the repository.
- **Branch:** Final code, `AUDIT_LOG.md`, and `REPORT.md` should be merged into the main branch before the cutoff.
- **Deadline:** 09.07.2026, 23:59 German time.
