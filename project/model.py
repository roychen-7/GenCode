import torch
import torch.nn as nn
import math


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting
    
    Paper: iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024)
    
    Key innovation: Inverts the traditional Transformer architecture by:
    - Embedding each variate (time series) as a token (not each time step)
    - Applying attention on variate dimension for multivariate correlations
    - Using FFN on temporal dimension for series representations
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract config parameters
        self.n_variates = config['n_variates']  # Number of variates (N)
        self.lookback_window = config['lookback_window']  # Input length (T)
        self.pred_window = config['pred_window']  # Prediction length (S)
        self.hidden_dim = config['hidden_dim']  # Token dimension (D)
        self.num_layers = config['num_layers']  # Number of Transformer blocks (L)
        self.num_heads = config['num_heads']  # Number of attention heads
        self.ff_dim = config['ff_dim']  # Feed-forward dimension
        self.dropout = config['dropout']
        
        # Embedding: project each time series from T -> D
        self.embedding = nn.Linear(self.lookback_window, self.hidden_dim)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Projection: project from D -> S for each variate
        self.projection = nn.Linear(self.hidden_dim, self.pred_window)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, lookback_window, n_variates)
               In traditional format: (B, T, N)
        
        Returns:
            predictions: Tensor of shape (batch_size, pred_window, n_variates)
                        In traditional format: (B, S, N)
        """
        # Input shape: (B, T, N)
        batch_size = x.shape[0]
        
        # Step 1: Transpose to (B, N, T) - treat each variate as a token
        x = x.transpose(1, 2)  # (B, N, T)
        
        # Step 2: Embed each variate token from T -> D
        # Shape: (B, N, T) -> (B, N, D)
        h = self.embedding(x)  # (B, N, D)
        
        # Step 3: Apply Transformer blocks
        # Attention operates on N dimension (variate tokens)
        # FFN operates on D dimension (series representations)
        for block in self.transformer_blocks:
            h = block(h)  # (B, N, D)
        
        # Step 4: Project each token from D -> S
        # Shape: (B, N, D) -> (B, N, S)
        out = self.projection(h)  # (B, N, S)
        
        # Step 5: Transpose back to (B, S, N)
        out = out.transpose(1, 2)  # (B, S, N)
        
        return out


class TransformerBlock(nn.Module):
    """
    Single Transformer block with:
    - Layer normalization
    - Multi-head self-attention (on variate dimension)
    - Feed-forward network (on representation dimension)
    """
    def __init__(self, hidden_dim, num_heads, ff_dim, dropout):
        super().__init__()
        
        # Layer norm (applied on temporal/feature dimension)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Multi-head attention (applied on variate dimension)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, N, D)
               B = batch size
               N = number of variates (tokens)
               D = hidden dimension
        
        Returns:
            Tensor of shape (B, N, D)
        """
        # Self-attention with residual connection
        # Attention operates on N dimension (captures multivariate correlations)
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual connection
        # FFN operates on D dimension (learns series representations)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x


class Model(nn.Module):
    """
    Wrapper class for training compatibility
    """
    def __init__(self, config):
        super().__init__()
        self.model = iTransformer(config)
    
    def forward(self, x):
        """
        Args:
            x: Input features of shape (batch_size, input_dim)
               Where input_dim = lookback_window * n_variates (flattened)
               Or if already shaped: (batch_size, lookback_window, n_variates)
        
        Returns:
            Predictions of shape (batch_size, output_dim)
            Where output_dim = pred_window * n_variates (flattened)
        """
        # Handle flattened input
        if len(x.shape) == 2:
            batch_size = x.shape[0]
            # Reshape from (B, T*N) to (B, T, N)
            x = x.view(batch_size, self.model.lookback_window, self.model.n_variates)
        
        # Forward through iTransformer
        # Input: (B, T, N), Output: (B, S, N)
        out = self.model(x)
        
        # Flatten output from (B, S, N) to (B, S*N) for compatibility
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)
        
        return out


def compute_loss(predictions, targets):
    """
    Compute MSE loss for time series forecasting
    
    Args:
        predictions: Tensor of shape (batch_size, pred_window * n_variates)
        targets: Tensor of shape (batch_size, 1) or (batch_size, pred_window * n_variates)
    
    Returns:
        MSE loss
    """
    # Handle different target shapes
    if targets.shape[1] == 1:
        # If target is single value, compare with first prediction
        predictions = predictions[:, 0:1]
    
    criterion = nn.MSELoss()
    return criterion(predictions, targets)
