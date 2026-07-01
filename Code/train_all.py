"""
Train and evaluate every dataset/model run listed in config_all.json.

Run from the Code directory:
    python train_all.py
"""
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import models
from data import get_loaders
from fit import Trainer


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


def get_class_weights(train_loader, num_classes, device):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, labels in train_loader:
        labels = labels.squeeze(1).cpu()
        counts += torch.bincount(labels, minlength=num_classes).float()

    weights = counts.sum() / (num_classes * counts.clamp_min(1))
    return weights.to(device)


def merge_config(base_config, run_config):
    merged = {key: value for key, value in base_config.items() if key != "RUNS"}
    merged.update(run_config)
    return merged


def train_one_config(config, device):
    print("=" * 70)
    print(f"Dataset: {config['DATA']} | Model: {config['MODEL']} | Epochs: {config['EPOCHS']}")
    print("=" * 70)

    train_loader, val_loader, test_loader = get_loaders(
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"],
    )

    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"],
        num_classes=config["NUM_CLASSES"],
        drop_rate=config.get("DROP_RATE", 0.5),
    ).to(device)

    if config.get("CLASS_WEIGHTED_LOSS", False):
        class_weights = get_class_weights(train_loader, config["NUM_CLASSES"], device)
        print(f"Using class-weighted loss: {class_weights.detach().cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    start = time.perf_counter()
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])
    train_seconds = time.perf_counter() - start

    checkpoint_dir = Path(config.get("CHECKPOINT_DIR", "checkpoints"))
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{config['DATA']}_{config['MODEL']}.pth"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to {checkpoint_path}")

    metrics = evaluate_test_set(model, test_loader, device)
    print(
        f"Test accuracy: {metrics['accuracy']:.2f}% | "
        f"Macro F1: {metrics['f1_macro']:.4f}"
    )

    return {
        "dataset": config["DATA"],
        "model": config["MODEL"],
        "epochs": config["EPOCHS"],
        "accuracy": f"{metrics['accuracy']:.4f}",
        "precision_macro": f"{metrics['precision_macro']:.4f}",
        "recall_macro": f"{metrics['recall_macro']:.4f}",
        "f1_macro": f"{metrics['f1_macro']:.4f}",
        "train_seconds": f"{train_seconds:.2f}",
        "checkpoint_path": str(checkpoint_path),
    }


def write_results(rows, results_path):
    path = Path(results_path)
    path.parent.mkdir(exist_ok=True)
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

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote benchmark results to {path}")


def main():
    with open("config_all.json", "r") as f:
        config = json.load(f)

    device = get_device()
    print(f"Training executing on device: {device}")

    results = []
    for run_config in config["RUNS"]:
        full_config = merge_config(config, run_config)
        result = train_one_config(full_config, device)
        results.append(result)
        write_results(results, config["RESULTS_PATH"])


if __name__ == "__main__":
    main()
