import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, n_tokens, d_model = x.shape
        
        Q = self.W_q(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, n_tokens, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, n_tokens, d_model)
        out = self.W_o(out)
        
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class iTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, n_variates, d_model)
        # Attention over variate tokens
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward for series representations
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x

class Stockformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding: project lookback series to token dimension
        self.embedding = nn.Sequential(
            nn.Linear(config.lookback_window, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            iTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        
        # Projection: token representation to prediction
        self.projection = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.pred_window)
        )
        
    def forward(self, x):
        # x shape: (batch, lookback_window, n_variates)
        # Invert: treat each variate as a token
        x = x.transpose(1, 2)  # (batch, n_variates, lookback_window)
        
        # Embed each variate token
        x = self.embedding(x)  # (batch, n_variates, d_model)
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Project to predictions
        x = self.projection(x)  # (batch, n_variates, pred_window)
        
        # Transpose back to time-first format
        x = x.transpose(1, 2)  # (batch, pred_window, n_variates)
        
        return x

def compute_loss(pred, target):
    # MSE loss for regression
    return nn.functional.mse_loss(pred, target)
