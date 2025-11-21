# train.py
import argparse
import torch
import torch.optim as optim
from config import TrainConfig
from models.mlp import MLP
from datasets import load_cn_data_full

def train_fast_mode(config):
    """
    Fast mode: load real cn_data but only run ONE training step.
    """
    device = config.device

    model = MLP(config.input_dim, config.hidden_dim, config.output_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    loader = load_cn_data_full(batch_size=config.batch_size, limit_batches=1)

    model.train()
    criterion = torch.nn.MSELoss()

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred.squeeze(), y.squeeze())
        loss.backward()
        optimizer.step()
        print("FAST MODE step_loss=", loss.item())
        break  # Only 1 step

    # 写一个 fast_mode_OK 文件，用来让 Go 控制器判断成功
    with open("fast_mode_OK.txt", "w") as f:
        f.write("OK\n")

    return loss.item()


def main_train(config):
    """
    Full training (不用于 Claude 自动测试)
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
