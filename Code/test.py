"""
Smoke tests for model construction and output shapes.

Run from the Code directory:
    python test.py
"""
import torch

import models


DATASETS = {
    "cells": {"channels": 3, "num_classes": 8},
    "chest": {"channels": 1, "num_classes": 2},
    "lesions": {"channels": 3, "num_classes": 7},
    "orgs": {"channels": 1, "num_classes": 11},
    "organs": {"channels": 1, "num_classes": 11},
}

MODEL_NAMES = ("AlexNet", "VGG16", "ResNet18", "MiniResNet")
BATCH_SIZE = 2
IMAGE_SIZE = 64


def test_model_output_shapes():
    for dataset_name, dataset_config in DATASETS.items():
        channels = dataset_config["channels"]
        num_classes = dataset_config["num_classes"]
        inputs = torch.randn(BATCH_SIZE, channels, IMAGE_SIZE, IMAGE_SIZE)

        for model_name in MODEL_NAMES:
            model_class = getattr(models, model_name)
            model = model_class(in_channels=channels, num_classes=num_classes)
            model.eval()

            with torch.no_grad():
                outputs = model(inputs)

            expected_shape = (BATCH_SIZE, num_classes)
            actual_shape = tuple(outputs.shape)

            assert actual_shape == expected_shape, (
                f"{model_name} on {dataset_name} returned {actual_shape}; "
                f"expected {expected_shape}"
            )

            print(f"PASS {model_name:<10} {dataset_name:<7} output={actual_shape}")


if __name__ == "__main__":
    print("PyTorch:", torch.__version__)
    test_model_output_shapes()
    print("All model shape smoke tests passed.")
