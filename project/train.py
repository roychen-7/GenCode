# train.py
import argparse
import torch
import torch.optim as optim
import numpy as np
from config import TrainConfig, ModelConfig
from model import Stockformer, compute_loss

def train_fast_mode(config):
    """
    Fast mode: load synthetic data and run ONE training step for testing.
    """
    device = config.device

    # Create model config
    model_config = ModelConfig(config)

    # Initialize Stockformer model
    model = Stockformer(model_config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    model.train()

    # Generate synthetic data for fast testing
    # Input: [batch_size, T1, N, 362]
    batch_size = config.batch_size
    T1 = config.T1
    T2 = config.T2
    N = config.num_stocks

    # Create random input data
    x = torch.randn(batch_size, T1, N, 362).to(device)

    # Create random target data
    # Returns: [batch_size, T2, N]
    y_true = torch.randn(batch_size, T2, N).to(device)
    # Trend labels: [batch_size, T2, N] with values 0 or 1
    y_trend_true = torch.randint(0, 2, (batch_size, T2, N)).to(device)

    # Forward pass
    optimizer.zero_grad()
    y_pred, y_pred_low, p_pred, p_pred_low = model(x)

    # Compute loss
    total_loss, loss_reg, loss_cla = compute_loss(
        y_pred, y_pred_low, p_pred, p_pred_low,
        y_true, y_trend_true,
        lambda_weight=config.lambda_weight
    )

    # Backward pass
    total_loss.backward()
    optimizer.step()

    print(f"FAST MODE - Total Loss: {total_loss.item():.4f}, Reg Loss: {loss_reg.item():.4f}, Cla Loss: {loss_cla.item():.4f}")

    # Write success marker file
    with open("fast_mode_OK.txt", "w") as f:
        f.write("OK\n")

    return total_loss.item()


def main_train(config):
    """
    Full training (not used for Claude auto-testing)
    """
    raise NotImplementedError("Full training is disabled for auto-iteration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Run in fast 1-step mode")
    args = parser.parse_args()

    config = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")

    if args.fast:
        train_fast_mode(config)
    else:
        main_train(config)
