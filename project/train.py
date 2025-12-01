import argparse
import torch
import torch.optim as optim
import numpy as np
from config import TrainConfig, ModelConfig
from model import Stockformer, compute_loss
from datasets import load_full_dataset

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)  # (batch, lookback, n_variates)
        labels = labels.to(device)  # (batch, 1)
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features)  # (batch, pred_window, n_variates)
        
        # For stock return prediction, we typically predict a single aggregated value
        # Sum or average across variates, or use first variate
        pred_return = predictions.mean(dim=-1)  # (batch, pred_window)
        
        # Compute loss
        loss = compute_loss(pred_return, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Batch {batch_idx + 1}, Loss: {loss.item():.6f}")
    
    return total_loss / num_batches

def main():
    # Initialize configs
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    print("Loading dataset...")
    try:
        dataloader = load_full_dataset(
            root=train_config.data_root,
            start_date=train_config.start_date,
            end_date=train_config.end_date,
            batch_size=train_config.batch_size,
            shuffle=True
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using dummy data for demonstration...")
        # Create dummy data for testing
        dummy_features = torch.randn(1000, model_config.lookback_window, model_config.n_variates)
        dummy_labels = torch.randn(1000, 1)
        from torch.utils.data import TensorDataset, DataLoader
        dataset = TensorDataset(dummy_features, dummy_labels)
        dataloader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)
    
    print("Initializing model...")
    model = Stockformer(model_config)
    model = model.to(train_config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
    
    print(f"Training on device: {train_config.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(train_config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{train_config.num_epochs}")
        avg_loss = train_epoch(model, dataloader, optimizer, train_config.device)
        print(f"Average Loss: {avg_loss:.6f}")
    
    print("\nTraining completed!")
    torch.save(model.state_dict(), 'stockformer_checkpoint.pt')
    print("Model saved to stockformer_checkpoint.pt")

if __name__ == "__main__":
    main()
