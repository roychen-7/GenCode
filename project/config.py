import torch

class ModelConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 4
        self.d_ff = 2048
        self.dropout = 0.1
        
        # Data dimensions
        self.n_variates = 158  # Alpha158 features
        self.lookback_window = 96
        self.pred_window = 1
        
class TrainConfig:
    def __init__(self):
        # Training hyperparameters
        self.batch_size = 1024
        self.learning_rate = 1e-4
        self.num_epochs = 1 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Data paths
        self.data_root = 'project/data/cn_data'
        self.start_date = '2010-01-01'
        self.end_date = '2020-12-31'
