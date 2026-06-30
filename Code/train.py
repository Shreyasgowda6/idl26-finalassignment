"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import json

import torch
import torch.nn as nn
import torch.optim as optim
from data import get_loaders
import models
from fit import Trainer

def main():   
    with open("config.json", "r") as f:
        config = json.load(f)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    # FIXED: was only checking cuda/cpu -- never used Apple Silicon GPU (MPS) acceleration on Mac
    print(f"Training executing on device: {device}")

    train_loader, val_loader, _ = get_loaders(data=config["DATA"], data_path=config["DATA_PATH"], batch_size=config["BATCH_SIZE"])

    model_class = getattr(models, config["MODEL"])
    # model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=0.99, activation_str=None).to(device)
    model = model_class(in_channels=config["CHANNELS"], num_classes=config["NUM_CLASSES"], drop_rate=config.get("DROP_RATE", 0.5)).to(device)
    # FIXED: drop_rate=0.99 zeroed out 99% of neurons during training, crippling learning -> now configurable, defaults to 0.5
    # FIXED: activation_str=None removed -- no model class accepts this kwarg, it was silently swallowed by **kwargs and did nothing
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    trainer = Trainer(model, criterion, optimizer, device)
    trainer.fit(train_loader, val_loader, epochs=config["EPOCHS"])

    # Save trained model weights so evaluate.py can load them later
    import os
    save_path = f"checkpoints/{config['DATA']}_{config['MODEL']}.pth"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
    
    
    
