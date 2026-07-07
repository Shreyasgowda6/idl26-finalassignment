"""
MAI/IDL SS26 - Final Assignment Part 3: Transfer Learning
Compares training ResNet18 from scratch vs fine-tuning from
a pretrained orgs checkpoint on the tiny organs dataset.
"""
import time
import os
import csv
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from data import get_loaders
from models import ResNet18

DATASET      = "organs"
DATA_PATH    = "../data"
BATCH_SIZE   = 16       # smaller batch — tiny dataset
CHANNELS     = 1
NUM_CLASSES  = 11
EPOCHS       = 20       # more epochs since dataset is tiny
LR_SCRATCH   = 0.001
LR_FINETUNE  = 0.0001   # lower LR for fine-tuning -- don't overwrite pretrained weights too fast
PRETRAINED   = "checkpoints/orgs_ResNet18.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Transfer learning on device: {device}")
print(f"Dataset: {DATASET} ({CHANNELS}ch, {NUM_CLASSES} classes)")
print("=" * 60)


def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.squeeze(1)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
    accuracy = accuracy_score(all_labels, all_preds) * 100
    return accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader, epochs, lr, label):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    print(f"\n── {label} ──")
    best_val_acc = 0
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = (correct / total) * 100

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).squeeze(1)
                outputs = model(images)
                _, predicted = outputs.max(1)
                v_total += labels.size(0)
                v_correct += predicted.eq(labels).sum().item()
        val_acc = (v_correct / v_total) * 100
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        print(f"   Epoch [{epoch+1:02d}/{epochs}] "
              f"Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%")

    model.load_state_dict(best_state)
    print(f"   Best val acc: {best_val_acc:.2f}% at epoch {best_epoch}")
    return model


def load_backbone_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint = {
        key: value
        for key, value in checkpoint.items()
        if not key.startswith("classifier")
    }
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"\nLoaded backbone weights from {checkpoint_path}")
    print("Classifier head reinitialized for organs.")
    if unexpected_keys:
        print(f"Unexpected keys skipped: {unexpected_keys}")
    classifier_keys = [key for key in missing_keys if key.startswith("classifier")]
    if classifier_keys:
        print(f"Classifier keys trained from scratch: {classifier_keys}")


# ── Load data ────────────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = get_loaders(
    data=DATASET, data_path=DATA_PATH, batch_size=BATCH_SIZE
)
print(f"Train batches: {len(train_loader)} | "
      f"Val batches: {len(val_loader)} | "
      f"Test batches: {len(test_loader)}")

# ── Approach 1: Train from scratch ───────────────────────────────────────────
model_scratch = ResNet18(in_channels=CHANNELS, num_classes=NUM_CLASSES)
model_scratch = train_model(
    model_scratch, train_loader, val_loader,
    epochs=EPOCHS, lr=LR_SCRATCH,
    label="ResNet18 trained FROM SCRATCH on organs"
)
scratch_acc, scratch_preds, scratch_labels = evaluate_model(model_scratch, test_loader)

# ── Approach 2: Transfer learning from orgs checkpoint ───────────────────────
model_transfer = ResNet18(in_channels=CHANNELS, num_classes=NUM_CLASSES)

if not os.path.exists(PRETRAINED):
    print(f"\nERROR: Pretrained checkpoint not found at {PRETRAINED}")
    print("Please run train.py with orgs/ResNet18 config first.")
    exit(1)

load_backbone_weights(model_transfer, PRETRAINED)

model_transfer = train_model(
    model_transfer, train_loader, val_loader,
    epochs=EPOCHS, lr=LR_FINETUNE,
    label="ResNet18 FINE-TUNED from orgs checkpoint on organs"
)
transfer_acc, transfer_preds, transfer_labels = evaluate_model(model_transfer, test_loader)

# ── Final comparison ─────────────────────────────────────────────────────────
print("\n")
print("=" * 60)
print("TRANSFER LEARNING RESULTS — TEST SET")
print("=" * 60)
print(f"\n{'From Scratch accuracy:':<35} {scratch_acc:.2f}%")
print(f"{'Transfer Learning accuracy:':<35} {transfer_acc:.2f}%")
print(f"{'Improvement:':<35} {transfer_acc - scratch_acc:+.2f}%")

print("\n── From Scratch — per-class metrics ──")
print(classification_report(scratch_labels, scratch_preds, digits=4))

print("\n── Transfer Learning — per-class metrics ──")
print(classification_report(transfer_labels, transfer_preds, digits=4))

# Save transfer-learning comparison
output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "transfer_learning_results.csv")

scratch_report = classification_report(
    scratch_labels,
    scratch_preds,
    digits=4,
    output_dict=True,
    zero_division=0,
)
transfer_report = classification_report(
    transfer_labels,
    transfer_preds,
    digits=4,
    output_dict=True,
    zero_division=0,
)

rows = [
    {
        "dataset": DATASET,
        "model": "ResNet18",
        "training_mode": "scratch",
        "epochs": EPOCHS,
        "learning_rate": LR_SCRATCH,
        "checkpoint": "",
        "accuracy": f"{scratch_acc:.4f}",
        "macro_precision": f"{scratch_report['macro avg']['precision']:.4f}",
        "macro_recall": f"{scratch_report['macro avg']['recall']:.4f}",
        "macro_f1": f"{scratch_report['macro avg']['f1-score']:.4f}",
        "weighted_precision": f"{scratch_report['weighted avg']['precision']:.4f}",
        "weighted_recall": f"{scratch_report['weighted avg']['recall']:.4f}",
        "weighted_f1": f"{scratch_report['weighted avg']['f1-score']:.4f}",
    },
    {
        "dataset": DATASET,
        "model": "ResNet18",
        "training_mode": "transfer_from_orgs",
        "epochs": EPOCHS,
        "learning_rate": LR_FINETUNE,
        "checkpoint": PRETRAINED,
        "accuracy": f"{transfer_acc:.4f}",
        "macro_precision": f"{transfer_report['macro avg']['precision']:.4f}",
        "macro_recall": f"{transfer_report['macro avg']['recall']:.4f}",
        "macro_f1": f"{transfer_report['macro avg']['f1-score']:.4f}",
        "weighted_precision": f"{transfer_report['weighted avg']['precision']:.4f}",
        "weighted_recall": f"{transfer_report['weighted avg']['recall']:.4f}",
        "weighted_f1": f"{transfer_report['weighted avg']['f1-score']:.4f}",
    },
]

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Transfer-learning results saved to {csv_path}")

# Save transfer model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model_transfer.state_dict(), "checkpoints/organs_ResNet18_transfer.pth")
print("Transfer model saved to checkpoints/organs_ResNet18_transfer.pth")
