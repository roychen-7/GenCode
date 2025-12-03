import torch
import torch.optim as optim
import torch.nn.functional as F
from model import Model, compute_loss


def train_model(dataloader, device, config):
    model_cfg = config.model
    train_cfg = config.train
    data_cfg = config.data

    model = Model(model_cfg).to(device)
    

    
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0)
    )
    
    epochs = train_cfg["num_epochs"]
    losses = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, device)
        losses.append(avg_loss)
        print(f"Average Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "result.pt")
    print("Model saved to result.pt")

    return {
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

    return total_loss / len(dataloader)

