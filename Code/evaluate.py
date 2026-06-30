"""
MAI/IDL SS26 - Final assignment.
Evaluation script: loads a trained model and reports test set metrics.
"""
import json
import os
import torch
from sklearn.metrics import classification_report, accuracy_score
import models
from data import get_loaders


def evaluate():
    with open("config.json", "r") as f:
        config = json.load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Evaluating on device: {device}")

    # Load data -- only need the test loader
    _, _, test_loader = get_loaders(
        data=config["DATA"],
        data_path=config["DATA_PATH"],
        batch_size=config["BATCH_SIZE"]
    )

    # Build the same model architecture used during training
    model_class = getattr(models, config["MODEL"])
    model = model_class(
        in_channels=config["CHANNELS"],
        num_classes=config["NUM_CLASSES"],
        drop_rate=config.get("DROP_RATE", 0.5)
    ).to(device)

    # Load the saved weights from training
    checkpoint_path = f"checkpoints/{config['DATA']}_{config['MODEL']}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        print("Please run train.py first to train and save the model.")
        return

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
    print(f"Dataset : {config['DATA']}")
    print(f"Model   : {config['MODEL']}")
    print(f"Accuracy: {accuracy:.2f}%")
    print()
    print("Per-class metrics (Precision / Recall / F1):")
    print(classification_report(all_labels, all_preds, digits=4))


if __name__ == "__main__":
    evaluate()