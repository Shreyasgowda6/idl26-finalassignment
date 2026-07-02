"""
Profile the Green Initiative model comparison for Part 2.

Run from the Code directory:
    python green_profile.py
"""
import csv
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

import models
from data import get_loaders


RUNS = [
    {
        "DATA": "cells",
        "MODEL": "ResNet18",
        "CHANNELS": 3,
        "NUM_CLASSES": 8,
        "EPOCHS": 20,
    },
    {
        "DATA": "cells",
        "MODEL": "MiniResNet",
        "CHANNELS": 3,
        "NUM_CLASSES": 8,
        "EPOCHS": 20,
    },
]

BASE_CONFIG = {
    "DATA_PATH": "../data",
    "BATCH_SIZE": 32,
    "DROP_RATE": 0.5,
    "LEARNING_RATE": 0.001,
    "RESULTS_PATH": "../outputs/green_profile_results.csv",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0


def parameter_count(model):
    return sum(param.numel() for param in model.parameters())


def model_size_mb(model):
    total_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    total_bytes += sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return total_bytes / (1024 ** 2)


def evaluate_accuracy(model, test_loader, device):
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

    return accuracy_score(all_labels, all_preds) * 100


def train_with_best_validation(model, criterion, optimizer, train_loader, val_loader, epochs, device):
    best_val_acc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())

    print("\n Starting Training Routine...")
    print("-" * 50)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / total
        train_acc = (correct / total) * 100

        val_loss, val_acc = evaluate_validation(model, criterion, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())

        print(f"Epoch [{epoch+1:02d}/{epochs:02d}] | "
              f"Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}%")

    print("-" * 50)
    print(f"Training Complete! Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}")

    model.load_state_dict(best_state)
    return best_epoch, best_val_acc


def evaluate_validation(model, criterion, dataloader, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / total, (correct / total) * 100


def inference_latency_ms(model, test_loader, device):
    model.eval()
    total_samples = 0

    reset_peak_memory(device)
    sync_if_needed(device)
    start = time.perf_counter()

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            total_samples += images.size(0)

    sync_if_needed(device)
    elapsed = time.perf_counter() - start
    peak_infer_mb = peak_memory_mb(device)

    return (elapsed / total_samples) * 1000, peak_infer_mb


def run_profile(config, device):
    print("=" * 70)
    print(f"Green profile: {config['DATA']} | {config['MODEL']} | {config['EPOCHS']} epochs")
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
        drop_rate=config["DROP_RATE"],
    ).to(device)

    params = parameter_count(model)
    size_mb = model_size_mb(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    reset_peak_memory(device)
    sync_if_needed(device)
    start = time.perf_counter()
    best_epoch, best_val_acc = train_with_best_validation(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        config["EPOCHS"],
        device,
    )
    sync_if_needed(device)
    train_seconds = time.perf_counter() - start
    peak_train_mb = peak_memory_mb(device)

    accuracy = evaluate_accuracy(model, test_loader, device)
    latency_ms, peak_infer_mb = inference_latency_ms(model, test_loader, device)

    result = {
        "dataset": config["DATA"],
        "model": config["MODEL"],
        "epochs": config["EPOCHS"],
        "best_epoch": best_epoch,
        "best_val_accuracy": f"{best_val_acc:.4f}",
        "accuracy": f"{accuracy:.4f}",
        "parameters": params,
        "model_size_mb": f"{size_mb:.4f}",
        "train_seconds": f"{train_seconds:.2f}",
        "inference_ms_per_sample": f"{latency_ms:.4f}",
        "peak_train_memory_mb": f"{peak_train_mb:.2f}",
        "peak_inference_memory_mb": f"{peak_infer_mb:.2f}",
    }

    print(
        f"Accuracy: {accuracy:.2f}% | Params: {params:,} | "
        f"Size: {size_mb:.2f} MB | Train: {train_seconds:.2f}s | "
        f"Inference: {latency_ms:.4f} ms/sample"
    )
    return result


def write_results(rows, results_path):
    path = Path(results_path)
    path.parent.mkdir(exist_ok=True)

    fieldnames = [
        "dataset",
        "model",
        "epochs",
        "best_epoch",
        "best_val_accuracy",
        "accuracy",
        "parameters",
        "model_size_mb",
        "train_seconds",
        "inference_ms_per_sample",
        "peak_train_memory_mb",
        "peak_inference_memory_mb",
    ]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote green profile results to {path}")


def main():
    device = get_device()
    print(f"Green profiling executing on device: {device}")

    rows = []
    for run in RUNS:
        config = {**BASE_CONFIG, **run}
        rows.append(run_profile(config, device))
        write_results(rows, BASE_CONFIG["RESULTS_PATH"])


if __name__ == "__main__":
    main()
