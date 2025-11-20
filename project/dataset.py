import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_tiny_dataset(root="data/tiny", batch_size=32):
    """
    Loads a tiny synthetic dataset (pre-generated).
    """
    features = np.load(os.path.join(root, "features.npy"))
    labels = np.load(os.path.join(root, "labels.npy"))

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)

    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)

def load_full_dataset(config):
    """
    Placeholder for your real Alpha158 data loading.
    When you integrate with Qlib, replace this.
    """
    raise NotImplementedError(
        "Implement full Alpha158 dataset loading here."
    )
