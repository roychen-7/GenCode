import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        # TODO: Initialize model components based on config
       
        
    def forward(self, x):
        # TODO: Define forward pass
        return x

def compute_loss(preds, labels):
    return F.mse_loss(preds, labels)
