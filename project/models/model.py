import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]

class TransformerModel(nn.Module):
    """
    Transformer-based model for time-series forecasting on Alpha158 data.
    
    Architecture:
    - Input embedding layer to project 158 features to d_model dimensions
    - Positional encoding
    - Transformer encoder with multiple layers
    - Output projection to regression target
    """
    def __init__(self, input_dim=158, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1, output_dim=1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, output_dim)
        
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch_size, input_dim] or [batch_size, seq_len, input_dim]
        
        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        # Handle both 2D and 3D inputs
        if x.dim() == 2:
            # [batch_size, input_dim] -> [batch_size, 1, input_dim]
            x = x.unsqueeze(1)
        
        # Project input to d_model dimensions
        # [batch_size, seq_len, input_dim] -> [batch_size, seq_len, d_model]
        x = self.input_projection(x)
        x = x * math.sqrt(self.d_model)  # Scale by sqrt(d_model) as in original Transformer
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Pass through transformer encoder
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x)
        
        # Take the last time step (or mean pooling)
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        x = x[:, -1, :]  # Use last time step
        
        # Project to output dimension
        # [batch_size, d_model] -> [batch_size, output_dim]
        x = self.output_projection(x)
        
        return x

# Keep MLP for backward compatibility
class MLP(nn.Module):
    def __init__(self, input_dim=158, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
