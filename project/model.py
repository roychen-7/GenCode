import torch
import torch.nn as nn
import math


class iTransformer(nn.Module):
    """
    iTransformer: Inverted Transformer for Time Series Forecasting
    
    From paper: Embeds each time series (variate) as a token, applies attention
    on variate dimension for multivariate correlations, and feed-forward network
    on temporal dimension for series representations.
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.lookback_window = config["lookback_window"]
        self.num_variates = config["num_variates"]
        self.d_model = config["d_model"]
        self.num_layers = config["num_layers"]
        self.num_heads = config["num_heads"]
        self.d_ff = config["d_ff"]
        self.dropout = config["dropout"]
        self.prediction_length = config["prediction_length"]
        
        # Embedding: project each variate's time series (lookback_window) to d_model
        self.embedding = nn.Linear(self.lookback_window, self.d_model)
        
        # Stack of iTransformer blocks
        self.layers = nn.ModuleList([
            iTransformerBlock(
                d_model=self.d_model,
                num_heads=self.num_heads,
                d_ff=self.d_ff,
                dropout=self.dropout
            )
            for _ in range(self.num_layers)
        ])
        
        # Projection: project d_model back to prediction_length
        self.projection = nn.Linear(self.d_model, self.prediction_length)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, lookback_window, num_variates)
        Returns:
            out: (batch_size, prediction_length, num_variates)
        """
        # Transpose: (batch_size, lookback_window, num_variates) -> (batch_size, num_variates, lookback_window)
        x = x.transpose(1, 2)
        
        # Embed each variate token: (batch_size, num_variates, lookback_window) -> (batch_size, num_variates, d_model)
        x = self.embedding(x)
        
        # Pass through iTransformer blocks
        for layer in self.layers:
            x = layer(x)
        
        # Project to prediction: (batch_size, num_variates, d_model) -> (batch_size, num_variates, prediction_length)
        x = self.projection(x)
        
        # Transpose back: (batch_size, num_variates, prediction_length) -> (batch_size, prediction_length, num_variates)
        x = x.transpose(1, 2)
        
        return x


class iTransformerBlock(nn.Module):
    """
    Single iTransformer block:
    - LayerNorm + Multi-head Self-Attention (on variate dimension)
    - LayerNorm + Feed-Forward Network (on each variate token)
    
    From paper: Layer normalization is applied on temporal dimension (series representations),
    attention captures multivariate correlations, FFN learns series representations.
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_variates, d_model)
        Returns:
            x: (batch_size, num_variates, d_model)
        """
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        
        return x


class Model(nn.Module):
    """
    Wrapper model matching train.py interface
    """
    
    def __init__(self, config):
        super().__init__()
        self.model = iTransformer(config)
        
    def forward(self, x):
        return self.model(x)


def compute_loss(predictions, targets):
    """
    MSE loss for time series forecasting
    
    Args:
        predictions: (batch_size, prediction_length, num_variates)
        targets: (batch_size, num_variates) - next time step
    Returns:
        loss: scalar
    """
    # Take only the first prediction step to match target
    pred_first_step = predictions[:, 0, :]
    
    # Compute MSE
    loss = nn.MSELoss()(pred_first_step, targets.squeeze(-1))
    
    return loss
