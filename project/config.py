class Config:
    def __init__(self):
        # Data parameters
        self.input_dim = 158
        self.output_dim = 1
        self.limit_samples = None
        
        # Model parameters
        self.hidden_dim = 64
        self.num_layers = 2
        self.dropout = 0.1
        
        # Training parameters
        self.num_epochs = 100
        self.batch_size = 128
        self.learning_rate = 0.001
        self.weight_decay = 1e-5
        
        # Device
        self.device = 'cpu'
        
    def set_test_mode(self):
        """Configure for fast test mode"""
        self.num_epochs = 1
        self.batch_size = 32
        self.limit_samples = 512

config = Config()
