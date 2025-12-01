import torch
import torch.nn as nn
import math


class Model(nn.Module):
    """
    iTransformer implementation based on paper.md.
    
    Architecture:
    - Embeds each time series (variate) as a token
    - Self-attention operates on variate tokens to capture multivariate correlations
    - Feed-forward network learns series representations
    - Layer normalization applied on temporal dimension of each variate
    """
    
    def __init__(self, config):
        super(Model, self).__init__()
        
        # Extract config parameters
        self.input_dim = config["input_dim"]  # Number of features per variate
        self.lookback_window = config["lookback_window"]  # T: lookback length
        self.pred_window = config["pred_window"]  # S: prediction length
        self.hidden_dim = config["hidden_dim"]  # D: token dimension
        self.num_layers = config["num_layers"]  # L: number of transformer blocks
        self.dropout = config.get("dropout", 0.1)
        
        # Number of variates N is determined by input_dim
        # In the iTransformer paradigm, each variate becomes a token
        self.num_variates = self.input_dim
        
        # Embedding: Projects time series (T points) to token dimension D
        # Input: (N, T) -> Output: (N, D)
        self.embedding = nn.Sequential(
            nn.Linear(self.lookback_window, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Stack of iTransformer blocks
        self.blocks = nn.ModuleList([
            iTransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=8,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Projection: Projects token dimension D back to prediction length S
        # Input: (N, D) -> Output: (N, S)
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, self.pred_window),
            nn.Dropout(self.dropout)
        )
    
    def forward(self, x):
        """
        Forward pass of iTransformer.
        
        Args:
            x: Input tensor of shape (batch_size, lookback_window, num_variates)
               Note: Standard format is (B, T, N)
        
        Returns:
            predictions: Tensor of shape (batch_size, pred_window, num_variates)
                        Note: Standard format is (B, S, N)
        """
        batch_size = x.shape[0]
        
        # Transpose: (B, T, N) -> (B, N, T)
        # Each variate's time series becomes a token
        x = x.transpose(1, 2)  # (B, N, T)
        
        # Embedding: (B, N, T) -> (B, N, D)
        h = self.embedding(x)  # (B, N, D)
        
        # Pass through iTransformer blocks
        for block in self.blocks:
            h = block(h)  # (B, N, D)
        
        # Projection: (B, N, D) -> (B, N, S)
        out = self.projection(h)  # (B, N, S)
        
        # Transpose back: (B, N, S) -> (B, S, N)
        out = out.transpose(1, 2)  # (B, S, N)
        
        return out


class iTransformerBlock(nn.Module):
    """
    Single iTransformer block consisting of:
    1. Layer Normalization + Self-Attention on variate tokens
    2. Layer Normalization + Feed-Forward Network on series representations
    """
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(iTransformerBlock, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Layer normalization for attention (applied on temporal/feature dimension)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Multi-head self-attention on variate tokens
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for FFN
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network for series representations
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Forward pass through one iTransformer block.
        
        Args:
            x: Input tensor of shape (batch_size, num_variates, hidden_dim)
        
        Returns:
            Output tensor of shape (batch_size, num_variates, hidden_dim)
        """
        # Self-attention with residual connection
        # Attention operates on variate tokens to capture multivariate correlations
        normed = self.norm1(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out
        
        # Feed-forward network with residual connection
        # FFN operates on series representations of each variate token
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


def compute_loss(predictions, targets):
    """
    Compute MSE loss between predictions and targets.
    
    Args:
        predictions: Predicted values of shape (batch_size, pred_window, num_variates)
        targets: Ground truth values of shape (batch_size, pred_window, num_variates)
                 or (batch_size, num_variates) for single-step prediction
    
    Returns:
        loss: Mean squared error loss
    """
    # Handle different target shapes
    if targets.dim() == 2:
        # Single-step prediction: (B, N) -> (B, 1, N)
        targets = targets.unsqueeze(1)
    
    # Ensure predictions match target shape
    if predictions.shape[1] != targets.shape[1]:
        # Take only the first step if multi-step prediction
        predictions = predictions[:, :targets.shape[1], :]
    
    # MSE loss
    loss = nn.functional.mse_loss(predictions, targets)
    
    return loss
