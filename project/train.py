import json
from typing import Dict, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import Stockformer, compute_loss


############################################################
# Data Interface Helpers
############################################################

def prepare_batch(
    X_panel: torch.Tensor,
    Y_true: torch.Tensor,
    time_slots: torch.Tensor,
    num_stocks: int,
    lookback_window: int,
    predict_window: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate and return batch tensors in the format expected by Stockformer.

    This function assumes that the dataset already constructs panel data in
    accordance with the paper's definition (Sec. 3.1 and Sec. 6.2.1):

        - X_panel \in R^{B×T1×N×362}
            channel 0: return at time t
            channel 1: trend indicator (0/1)
            channels 2..361: 360 Alpha360 factors
        - Y_true \in R^{B×T2×N×2}
            [...,0]: future returns
            [...,1]: trend labels (0/1) for the same horizons
        - time_slots \in R^{B×T1}
            integer time slot indices for each historical time step.

    No fabrication or broadcasting of features is done here; the caller is
    responsible for building proper panel tensors from raw market data.

    Args:
        X_panel: (B,T1,N,362) input panel.
        Y_true:  (B,T2,N,2) targets.
        time_slots: (B,T1) time-slot indices.
        num_stocks: N.
        lookback_window: T1.
        predict_window: T2.

    Returns:
        X_panel, Y_true, time_slots in the same shape, after basic checks.
    """
    if X_panel.dim() != 4:
        raise ValueError(f"X_panel must be 4D, got {tuple(X_panel.shape)}")
    if Y_true.dim() != 4:
        raise ValueError(f"Y_true must be 4D, got {tuple(Y_true.shape)}")
    if time_slots.dim() != 2:
        raise ValueError(f"time_slots must be 2D, got {tuple(time_slots.shape)}")

    B, T1, N, F = X_panel.shape
    B_y, T2, N_y, F_y = Y_true.shape
    B_t, T1_t = time_slots.shape

    if B_y != B or B_t != B:
        raise ValueError("Batch size mismatch among X_panel, Y_true, and time_slots")
    if N != num_stocks or N_y != num_stocks:
        raise ValueError(f"Number of stocks mismatch: expected {num_stocks}, got {N} and {N_y}")
    if T1 != lookback_window or T1_t != lookback_window:
        raise ValueError(
            f"Lookback length mismatch: expected {lookback_window}, got {T1} and {T1_t}"
        )
    if T2 != predict_window:
        raise ValueError(f"Predict window mismatch: expected {predict_window}, got {T2}")
    if F != 362:
        raise ValueError(f"X_panel last dimension must be 362, got {F}")
    if F_y != 2:
        raise ValueError(f"Y_true last dimension must be 2 (return, trend), got {F_y}")

    return X_panel, Y_true, time_slots


############################################################
# Training Loop (Sec. 6.2.4)
############################################################

def train_model(
    dataloader: DataLoader,
    device: str,
    config,
) -> Dict:
    """Train the Stockformer model according to the paper's training setup.

    Expected DataLoader output per batch:
        batch = (X_panel, Y_true, time_slots)
        - X_panel:  (B, T1, N, 362)
        - Y_true:   (B, T2, N, 2)
        - time_slots:(B, T1)

    Config structure (compatible with config_paper.json):
        config.model: {
            "hidden_dim": 128,
            "num_heads": 1,
            "num_layers": 2,
            "dropout": 0.2,
            "lookback_window": 20,
            "predict_window": 2,
            "num_stocks": 300,
            "kernel_size": 2,
            ...
        }
        config.train: {
            "batch_size": 2,
            "num_epochs": 100,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "lr_decay_step": 30,
            "lr_decay_rate": 0.1,
            "lambda_cla": 2.0
        }

    The optimizer, learning rate schedule, and hyperparameters follow Sec. 6.2.4:
        - Adam optimizer
        - initial lr = 0.001
        - StepLR with decay rate 0.1
        - dropout = 0.2
        - batch_size = 2
        - num_epochs = 100

    Returns:
        summary dict with final loss, per-epoch losses, and training configuration.
    """

    device_t = torch.device(device)

    model_cfg = config.model
    train_cfg = config.train

    lookback_window = int(model_cfg.get("lookback_window", 20))
    predict_window = int(model_cfg.get("predict_window", 2))
    num_stocks = int(model_cfg.get("num_stocks", 300))

    # Initialize Stockformer
    model = Stockformer(model_cfg).to(device_t)

    # Optimizer and LR scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(train_cfg.get("lr_decay_step", 30)),
        gamma=float(train_cfg.get("lr_decay_rate", 0.1)),
    )

    num_epochs = int(train_cfg.get("num_epochs", 100))
    lambda_cla = float(train_cfg.get("lambda_cla", 2.0))

    model.train()
    epoch_losses = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            # Unpack batch; dataset should provide panel tensors directly
            if len(batch) == 3:
                X_panel, Y_true, time_slots = batch
            else:
                raise ValueError(
                    "Dataloader must yield (X_panel, Y_true, time_slots) per batch"
                )

            X_panel = X_panel.to(device_t)
            Y_true = Y_true.to(device_t)
            time_slots = time_slots.to(device_t)

            # Basic shape validation
            X_panel, Y_true, time_slots = prepare_batch(
                X_panel,
                Y_true,
                time_slots,
                num_stocks=num_stocks,
                lookback_window=lookback_window,
                predict_window=predict_window,
            )

            # Forward pass
            outputs = model(X_panel, time_slots)

            # Compute multitask loss (Eq. (12)-(14))
            loss = compute_loss(outputs, Y_true, lambda_cla=lambda_cla)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        scheduler.step()

        avg_loss = running_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Save trained weights
    torch.save(model.state_dict(), "result.pt")

    return {
        "final_loss": epoch_losses[-1] if epoch_losses else 0.0,
        "epochs": num_epochs,
        "losses": epoch_losses,
        "device": str(device_t),
    }

