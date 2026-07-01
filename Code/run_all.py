"""
Train and evaluate every required dataset/model combination.

Run from the Code directory:
    python run_all.py
"""
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import models
from data import get_loaders
from fit import Trainer


DATASETS = {
    "cells": {"channels": 3, "num_classes": 8},
    "chest": {"channels": 1, "num_classes": 2},
    "lesions": {"channels": 3, "num_classes": 7},
    "orgs": {"channels": 1, "num_classes": 11},
}

MODEL_NAMES = ("AlexNet", "VGG16", "ResNet18")

DATA_PATH = "data"
BATCH_SIZE = 32
DROP_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 20
RESULTS_PATH = Path("../outputs/benchmark_results.csv")
CHECKPOINT_DIR = Path("checkpoints")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_test_set(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def run_single_experiment(dataset_name, dataset_config, model_name, device):
    print("=" * 70)
    print(f"Dataset: {dataset_name} | Model: {model_name} | Epochs: {EPOCHS}")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_loaders(
        data=dataset_name,
        data_path=DATA_PATH,
        batch_size=BATCH_SIZE,
    )

    model_class = getattr(models, model_name)
    model = model_class(
        in_channels=dataset_config["channels"],
        num_classes=dataset_config["num_classes"],
        drop_rate=DROP_RATE,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    trainer = Trainer(model, criterion, optimizer, device)

    start = time.perf_counter()
    trainer.fit(train_loader, val_loader, epochs=EPOCHS)
    train_seconds = time.perf_counter() - start

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{dataset_name}_{model_name}.pth"
    torch.save(model.state_dict(), checkpoint_path)

    metrics = evaluate_test_set(model, test_loader, device)
    print(
        f"Test accuracy: {metrics['accuracy']:.2f}% | "
        f"Macro F1: {metrics['f1_macro']:.4f}"
    )
    print(f"Saved checkpoint: {checkpoint_path}")

    return {
        "dataset": dataset_name,
        "model": model_name,
        "epochs": EPOCHS,
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision_macro": f"{metrics['precision_macro']:.4f}",
        "recall_macro": f"{metrics['recall_macro']:.4f}",
        "f1_macro": f"{metrics['f1_macro']:.4f}",
        "train_seconds": f"{train_seconds:.2f}",
        "checkpoint_path": str(checkpoint_path),
    }


def write_results(rows):
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "epochs",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "train_seconds",
        "checkpoint_path",
    ]

    with RESULTS_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote benchmark results to {RESULTS_PATH}")


def main():
    device = get_device()
    print(f"Running benchmark suite on device: {device}")

    rows = []
    for dataset_name, dataset_config in DATASETS.items():
        for model_name in MODEL_NAMES:
            row = run_single_experiment(dataset_name, dataset_config, model_name, device)
            rows.append(row)
            write_results(rows)

    print("\nAll benchmark runs completed.")


if __name__ == "__main__":
    main()
