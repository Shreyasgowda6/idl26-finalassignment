"""
MAI/IDL SS26 - Final assignment.
Evaluation script: loads a trained model and reports test set metrics.
"""
import json
import os
import csv
import argparse
import torch
from sklearn.metrics import classification_report, accuracy_score
import models
from data import get_loaders


DATASET_SPECS = {
    "cells": {"channels": 3, "num_classes": 8},
    "chest": {"channels": 1, "num_classes": 2},
    "lesions": {"channels": 3, "num_classes": 7},
    "orgs": {"channels": 1, "num_classes": 11},
}


def get_device():
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def merge_config(base_config, run_config):
    merged = {key: value for key, value in base_config.items() if key != "RUNS"}
    merged.update(run_config)
    return merged


def get_output_path():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, "evaluation_results.csv")


def write_results(rows, csv_path):
    if not rows:
        return

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote evaluation results to {csv_path}")


def evaluate_one(config, device):
    dataset = config["DATA"]
    if dataset not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset '{dataset}'. Expected one of {list(DATASET_SPECS)}")

    dataset_spec = DATASET_SPECS[dataset]
    if (
        config.get("CHANNELS") != dataset_spec["channels"]
        or config.get("NUM_CLASSES") != dataset_spec["num_classes"]
    ):
        print(
            "Config dataset metadata does not match DATA. "
            f"Using CHANNELS={dataset_spec['channels']} and "
            f"NUM_CLASSES={dataset_spec['num_classes']} for {dataset}."
        )

    # Load data -- only need the test loader
    _, _, test_loader = get_loaders(
        data=dataset,
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"]
    )

    # Build the same model architecture used during training
    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=dataset_spec["channels"],
        num_classes=dataset_spec["num_classes"],
        drop_rate=config.get("DROP_RATE", 0.5)
    ).to(device)

    # Load the saved weights from training
    checkpoint_path = f"checkpoints/{dataset}_{config['MODEL']}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please run train.py first to train and save the model.")
        return None

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"Loaded weights from {checkpoint_path}")
    print("-" * 50)

    # Run inference on the full test set
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze(1)  # (N, 1) -> (N,)

            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Compute and print metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    report = classification_report(
        all_labels,
        all_preds,
        digits=4,
        output_dict=True
    )
    print(f"Dataset : {dataset}")
    print(f"Model   : {config['MODEL']}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("Per-class metrics (Precision / Recall / F1):")
    print(classification_report(all_labels, all_preds, digits=4))

    return {
        "dataset": dataset,
        "model": config["MODEL"],
        "epochs": config.get("EPOCHS", ""),
        "checkpoint": checkpoint_path,
        "accuracy": f"{accuracy:.4f}",
        "macro_precision": f"{report['macro avg']['precision']:.4f}",
        "macro_recall": f"{report['macro avg']['recall']:.4f}",
        "macro_f1": f"{report['macro avg']['f1-score']:.4f}",
        "weighted_precision": f"{report['weighted avg']['precision']:.4f}",
        "weighted_recall": f"{report['weighted avg']['recall']:.4f}",
        "weighted_f1": f"{report['weighted avg']['f1-score']:.4f}",
    }


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoints.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate every run in config_all.json instead of only config.json.",
    )
    args = parser.parse_args()

    config_file = "config_all.json" if args.all else "config.json"
    with open(config_file, "r") as f:
        config = json.load(f)

    device = get_device()
    print(f"Evaluating on device: {device}")

    if args.all:
        rows = []
        for run_config in config["RUNS"]:
            full_config = merge_config(config, run_config)
            print("=" * 70)
            print(f"Dataset: {full_config['DATA']} | Model: {full_config['MODEL']}")
            print("=" * 70)
            row = evaluate_one(full_config, device)
            if row is not None:
                rows.append(row)
                write_results(rows, get_output_path())
        return

    row = evaluate_one(config, device)
    if row is None:
        return

    csv_path = get_output_path()
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            existing_rows = list(csv.DictReader(f))

    existing_rows.append(row)
    write_results(existing_rows, csv_path)


if __name__ == "__main__":
    evaluate()
