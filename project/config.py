from dataclasses import dataclass

@dataclass
class TrainConfig:
    input_dim: int = 158          # Alpha158 feature dimension
    hidden_dim: int = 64          # d_model for Transformer
    output_dim: int = 1           # Regression target
    
    # Transformer-specific parameters
    nhead: int = 4                # Number of attention heads
    num_layers: int = 2           # Number of transformer encoder layers
    dim_feedforward: int = 256    # Dimension of feedforward network
    dropout: float = 0.1          # Dropout rate
    
    # Training parameters
    lr: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 1
    device: str = "cpu"
