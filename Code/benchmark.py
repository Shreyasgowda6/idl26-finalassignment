"""
MAI/IDL SS26 - Final Assignment Part 2: Green/Efficient Model Benchmark
Compares ResNet18 vs MiniResNet on training time, inference speed,
parameter count, memory usage, and accuracy.
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
from models import ResNet18, MiniResNet

DATASET     = "cells"
DATA_PATH   = "../data"
BATCH_SIZE  = 32
CHANNELS    = 3
NUM_CLASSES = 8
EPOCHS      = 10
LR          = 0.001

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Benchmarking on device: {device}")
print("=" * 60)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_size_mb(model):
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_inference(model, loader, n_batches=50):
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            imgs, _ = next(iter(loader))
            _ = model(imgs.to(device))
    start = time.perf_counter()
    with torch.no_grad():
        for i, (imgs, _) in enumerate(loader):
            if i >= n_batches:
                break
            _ = model(imgs.to(device))
    elapsed = time.perf_counter() - start
    total_images = min(n_batches, len(loader)) * BATCH_SIZE
    return (elapsed / total_images) * 1000


def train_and_benchmark(model, train_loader, val_loader, label):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    model.to(device)
    epoch_times = []
    final_val_acc = 0

    print(f"\n── {label} ──")
    print(f"   Parameters : {count_parameters(model):,}")
    print(f"   Model size : {model_size_mb(model):.2f} MB")

    for epoch in range(EPOCHS):
        model.train()
        correct, total = 0, 0
        start = time.perf_counter()

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

        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
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

        print(f"   Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Acc: {train_acc:.2f}%  "
              f"Val Acc: {val_acc:.2f}%  "
              f"Time: {epoch_time:.1f}s")
        final_val_acc = val_acc

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    ms_per_image = benchmark_inference(model, val_loader)

    print(f"\n   Avg epoch time  : {avg_epoch_time:.2f}s")
    print(f"   Inference speed : {ms_per_image:.3f} ms/image")
    print(f"   Final val acc   : {final_val_acc:.2f}%")

    return {
        "params": count_parameters(model),
        "size_mb": model_size_mb(model),
        "avg_epoch_time": avg_epoch_time,
        "ms_per_image": ms_per_image,
        "final_val_acc": final_val_acc,
    }


train_loader, val_loader, _ = get_loaders(
    data=DATASET, data_path=DATA_PATH, batch_size=BATCH_SIZE
)

r = train_and_benchmark(
    ResNet18(in_channels=CHANNELS, num_classes=NUM_CLASSES),
    train_loader, val_loader, label="ResNet18 (original)"
)

m = train_and_benchmark(
    MiniResNet(in_channels=CHANNELS, num_classes=NUM_CLASSES),
    train_loader, val_loader, label="MiniResNet (efficient)"
)

print("\n")
print("=" * 60)
print("BENCHMARK COMPARISON SUMMARY")
print("=" * 60)
print(f"{'Metric':<25} {'ResNet18':>15} {'MiniResNet':>15} {'Reduction':>12}")
print("-" * 60)

def pct_reduction(a, b):
    return f"{((a - b) / a * 100):.1f}%"

print(f"{'Parameters':<25} {r['params']:>15,} {m['params']:>15,} {pct_reduction(r['params'], m['params']):>12}")
print(f"{'Model size (MB)':<25} {r['size_mb']:>14.2f}MB {m['size_mb']:>14.2f}MB {pct_reduction(r['size_mb'], m['size_mb']):>12}")
print(f"{'Avg epoch (s)':<25} {r['avg_epoch_time']:>14.2f}s {m['avg_epoch_time']:>14.2f}s {pct_reduction(r['avg_epoch_time'], m['avg_epoch_time']):>12}")
print(f"{'Inference (ms/img)':<25} {r['ms_per_image']:>13.3f}ms {m['ms_per_image']:>13.3f}ms {pct_reduction(r['ms_per_image'], m['ms_per_image']):>12}")

acc_diff = r["final_val_acc"] - m["final_val_acc"]
print(f"{'Val accuracy':<25} {r['final_val_acc']:>14.2f}% {m['final_val_acc']:>14.2f}% {f'-{acc_diff:.2f}%':>12}")
print("=" * 60)
