import torch
import torch.optim as optim
import argparse
import json

from config import FinalConfig
from model import Model, compute_loss
from datasets import load_full_dataset


# --------------------------
# Auto device selection
# --------------------------
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


# --------------------------
# One epoch training
# --------------------------
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(features)
        loss = compute_loss(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx+1}, Loss: {loss.item():.6f}")

    return total_loss / len(dataloader)


# --------------------------
# Main training entry
# --------------------------
def main():
    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode (force epochs=1)"
    )
    args = parser.parse_args()

    # Load final config (already validated & merged)
    cfg = FinalConfig()

    model_cfg = cfg.model
    train_cfg = cfg.train
    data_cfg = cfg.data

    # Determine device
    device = get_device()
    print(f"Auto-selected device: {device}")
    train_cfg["device"] = device

    # Test mode
    if args.test:
        print(">>> Running in TEST MODE (epochs = 1)")
        train_cfg["num_epochs"] = 1

    # Load dataset
    print("Loading dataset...")
    dataloader = load_full_dataset(
        instruments=data_cfg["instruments"] if "instruments" in data_cfg else "csi300",
        features=data_cfg["features"],
        label=data_cfg["label"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        lookback_window=model_cfg["lookback_window"]
    )

    print(model_cfg)

    # Initialize model
    print("Initializing model...")
    model = Model(model_cfg).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    epochs = train_cfg["num_epochs"]
    losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        losses.append(avg_loss)
        print(f"Average Loss: {avg_loss:.6f}")

    # Save trained model
    torch.save(model.state_dict(), "result.pt")
    print("Model saved to result.pt")

    # Save summary
    summary = {
        "model_name": model.__class__.__name__,
        "num_params": sum(p.numel() for p in model.parameters()),
        "epochs": epochs,
        "final_loss": losses[-1],
        "best_loss": min(losses),
        "best_epoch": int(losses.index(min(losses)) + 1),
        "device": device,
        "result_file": "result.pt",
        "config_used": {
            "model": model_cfg,
            "train": train_cfg,
            "data": data_cfg
        }
    }

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Summary saved to summary.json")


if __name__ == "__main__":
    main()
