# IDL26 Final Assignment

## Project Summary

This repository contains a corrected medical image classification pipeline for the IDL26 final assignment. The work includes code repair, configuration-driven training, benchmark evaluation across four datasets, an efficient model analysis, and a transfer-learning experiment for scarce data.

The main benchmark evaluates three CNN architectures:

- AlexNet
- VGG16
- ResNet18

across four medical imaging datasets:

- `cells`
- `chest`
- `lesions`
- `orgs`

## Repository Structure

```text
Code/
  data.py            Dataset loading and train/validation/test splitting
  fit.py             Training and validation loop
  models.py          AlexNet, VGG16, ResNet18, and MiniResNet definitions
  train.py           Single-run training entry point
  evaluate.py        Test-set evaluation script
  train_all.py       Optional benchmark runner for config_all.json
  config.json        Single-run configuration
  config_all.json    Full benchmark configuration

AUDIT_LOG.md         Bug audit and fix explanations
REPORT.md            Final benchmark report and analysis
assignment_final.pdf Assignment specification
```

Generated files such as datasets, checkpoints, local virtual environments, and benchmark outputs are intentionally ignored by Git.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
pip install torch torchvision numpy scikit-learn
```

Place the dataset files in a top-level `data/` directory:

```text
data/
  cells.pt
  chest.pt
  lesions.pt
  orgs.pt
  organs.pt
```

## Training

Edit `Code/config.json`, then run from inside `Code/`:

```bash
python train.py
```

Example:

```json
{
  "DATA": "orgs",
  "DATA_PATH": "../data",
  "BATCH_SIZE": 32,
  "MODEL": "ResNet18",
  "CHANNELS": 1,
  "NUM_CLASSES": 11,
  "DROP_RATE": 0.5,
  "LEARNING_RATE": 0.001,
  "EPOCHS": 20
}
```

Trained checkpoints are written to:

```text
Code/checkpoints/{DATA}_{MODEL}.pth
```

## Evaluation

After training a model, run:

```bash
python evaluate.py
```

The evaluation script reports test accuracy, per-class precision/recall/F1, and macro averages.

## Final Benchmark Results

All final benchmark runs exceeded the required accuracy thresholds.

| Dataset | Best Model by Accuracy | Accuracy | Macro F1 | Target |
|---|---|---:|---:|---:|
| cells | ResNet18 | 97.16% | 0.9719 | 90% |
| chest | ResNet18 | 90.06% | 0.8890 | 87% |
| lesions | ResNet18 | 76.61% | 0.4834 | 67% |
| orgs | VGG16 | 90.06% | 0.8857 | 83% |

Full results for all 12 dataset/model combinations are available in `REPORT.md`.

## Audit Summary

The recovered pipeline contained crash bugs, silent model bugs, and training-loop issues. The main fixes include:

- Correct dataset filename loading and train/validation splitting.
- Fixed VGG channel propagation inside repeated convolution blocks.
- Fixed AlexNet input channel, output class, and classifier dimension handling.
- Fixed ResNet18 activation and classifier output behavior.
- Removed unsafe in-place residual addition.
- Restored stable training behavior by fixing gradient clearing, label shape handling, and dropout configuration.

Detailed explanations are available in `AUDIT_LOG.md`.
