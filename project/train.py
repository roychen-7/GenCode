import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from config import config
from models.model import Model
from datasets import load_full_dataset

def prepare_data(mode='full'):
    """Load and prepare dataset"""
    data = load_full_dataset()
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_valid = data['X_valid']
    y_valid = data['y_valid']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Apply limit_samples in test mode
    if mode == 'test' and config.limit_samples is not None:
        limit = config.limit_samples
        X_train = X_train[:limit]
        y_train = y_train[:limit]
        X_valid = X_valid[:min(limit // 4, len(X_valid))]
        y_valid = y_valid[:min(limit // 4, len(y_valid))]
        X_test = X_test[:min(limit // 4, len(X_test))]
        y_test = y_test[:min(limit // 4, len(y_test))]
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
    X_valid_tensor = torch.FloatTensor(X_valid)
    y_valid_tensor = torch.FloatTensor(y_valid).unsqueeze(1)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def evaluate(model, data_loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='full', choices=['full', 'test'],
                        help='Training mode: full or test')
    args = parser.parse_args()
    
    # Configure for test mode if needed
    if args.mode == 'test':
        config.set_test_mode()
    
    # Set device
    device = torch.device(config.device)
    
    # Load data
    print(f"Loading data in {args.mode} mode...")
    train_loader, valid_loader, test_loader = prepare_data(mode=args.mode)
    print(f"Train batches: {len(train_loader)}, Valid batches: {len(valid_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = Model(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    for epoch in range(config.num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        
        if args.mode == 'full' or epoch == 0:
            print(f"Epoch {epoch+1}/{config.num_epochs} - Train Loss: {train_loss:.6f}, Valid Loss: {valid_loss:.6f}")
    
    # Final evaluation on test set
    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.6f}")
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
