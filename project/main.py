import torch
import argparse
import json

from config import FinalConfig
from load_data import load_full_dataset
from train import train_model


# --------------------------
# Auto device selection
# --------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

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

    # Determine device
    device = get_device()
    print(f"Auto-selected device: {device}")
    cfg.train["device"] = device

    # Test mode
    if args.test:
        print(">>> Running in TEST MODE (epochs = 1)")
        cfg.train["num_epochs"] = 1

    # Load dataset
    print("Loading dataset...")
    dataloader = load_full_dataset(
        instruments=cfg.data["instruments"] if "instruments" in cfg.data else "csi300",
        features=cfg.data["features"],
        label=cfg.data["label"],
        start_date=cfg.data["start_date"],
        end_date=cfg.data["end_date"],
        batch_size=cfg.train["batch_size"],
        shuffle=True,
        lookback_window=cfg.model["lookback_window"]
    )

    summary = train_model(dataloader, device, cfg)

    with open("summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("Summary saved to summary.json")

if __name__ == "__main__":
    main()
