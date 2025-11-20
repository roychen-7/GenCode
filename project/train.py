import argparse
import torch
import torch.optim as optim

from config import TrainConfig
from models.model import TransformerModel, MLP
from dataset import load_tiny_dataset, load_full_dataset

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = torch.nn.MSELoss()

    for batch in loader:
        x, y = batch
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def training_loop(config: TrainConfig, mode="full", model_type="transformer"):
    device = config.device

    if model_type == "transformer":
        model = TransformerModel(
            input_dim=config.input_dim,
            d_model=config.hidden_dim,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            output_dim=config.output_dim,
        ).to(device)
    else:
        model = MLP(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    if mode == "test":
        loader = load_tiny_dataset(batch_size=config.batch_size)
        config.num_epochs = 1  # tiny mode: fast
    else:
        loader = load_full_dataset(config)

    history = []

    for epoch in range(config.num_epochs):
        loss = train_one_epoch(model, loader, optimizer, device)
        history.append(loss)

    # tiny 模式将 log 写到文件，pytest 会读取它
    if mode == "test":
        import json
        with open("tiny_train_log.json", "w") as f:
            json.dump({"loss_history": history}, f)

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="full")
    parser.add_argument("--model", choices=["transformer", "mlp"], default="transformer")
    args = parser.parse_args()

    config = TrainConfig(device="cuda" if torch.cuda.is_available() else "cpu")
    training_loop(config, mode=args.mode, model_type=args.model)
