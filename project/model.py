import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


############################################################
# Utility
############################################################

def _check_shape(x: torch.Tensor, dims: int, name: str):
    if x.dim() != dims:
        raise ValueError(f"{name} must be {dims}D tensor, got shape {tuple(x.shape)}")


############################################################
# Wavelet-based Decoupling Flow Layer (Sec. 4.2, Eq. (3)-(4))
############################################################

class DecouplingFlowLayer(nn.Module):
    """Decoupling Flow Layer implementing a single-level DWT on the return channel.

    Paper mapping:
    - Eq. (3): X_l = g X, X_h = h X (low-/high-pass filtering + downsampling).
    - Eq. (4): X_l = W_g g^T X_l + b_g, X_h = W_h h^T X_h + b_h (upsample and project).

    This implementation:
    - Applies Daubechies-4 analysis filters (dec_lo, dec_hi) on the return series
      along the temporal dimension using Conv1d with stride 2 (downsampling).
    - Uses corresponding synthesis filters via ConvTranspose1d with stride 2 to
      upsample back to original temporal length.
    - ✅ g / h filters are now LEARNABLE (weights initialized from db4 but not frozen),
      in line with the paper's "learned convolution kernels".
    - Concatenates upsampled low- or high-frequency return with unchanged trend
      and 360 Alpha360 factors, then applies separate linear projections Wg, Wh.

    Input:
        X: (B, T1, N, 362)
            channel 0: return r_t
            channel 1: trend indicator (0/1)
            channels 2..361: 360 Alpha360 factors
    Output:
        X_l, X_h: (B, T1, N, D)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        # Total feature channels per stock per time: 1 return + 1 trend + 360 factors
        self.in_feat_total = 362

        # Daubechies-4 ('db4') analysis filter coefficients (dec_lo, dec_hi)
        dec_lo = [
            -0.0105974017850021,
            0.0328830116668852,
            0.0308413818359869,
            -0.1870348117188811,
            -0.0279837694169839,
            0.6308807679298587,
            0.7148465705529156,
            0.2303778133088552,
        ]
        dec_hi = [
            -0.2303778133088552,
            0.7148465705529156,
            -0.6308807679298587,
            -0.0279837694169839,
            0.1870348117188811,
            0.0308413818359869,
            -0.0328830116668852,
            -0.0105974017850021,
        ]

        k = len(dec_lo)
        self.kernel_size = k

        # 保存 filters 用于初始化
        self.register_buffer("g", torch.tensor(dec_lo, dtype=torch.float32).view(1, 1, k))
        self.register_buffer("h", torch.tensor(dec_hi, dtype=torch.float32).view(1, 1, k))

        # --------------------------------------------------
        # ✅ Analysis filters g, h (Conv1d, stride=2, LEARNABLE)
        # --------------------------------------------------
        self.dwt_low = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=2,
            padding=0,
            bias=True,
        )
        self.dwt_high = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=2,
            padding=0,
            bias=True,
        )
        with torch.no_grad():
            self.dwt_low.weight.copy_(self.g)
            self.dwt_high.weight.copy_(self.h)
        # 不再 requires_grad_(False)，保持可学习

        # --------------------------------------------------
        # ✅ Synthesis filters g^T, h^T (ConvTranspose1d, LEARNABLE)
        # --------------------------------------------------
        self.idwt_low = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=2,
            padding=0,
            bias=True,
        )
        self.idwt_high = nn.ConvTranspose1d(
            in_channels=1,
            out_channels=1,
            kernel_size=k,
            stride=2,
            padding=0,
            bias=True,
        )
        with torch.no_grad():
            self.idwt_low.weight.copy_(self.g)
            self.idwt_high.weight.copy_(self.h)

        # Learnable projections Wg, Wh (+ implicit bias)
        self.proj_low = nn.Linear(self.in_feat_total, self.hidden_dim)
        self.proj_high = nn.Linear(self.in_feat_total, self.hidden_dim)

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            X: (B, T1, N, 362)

        Returns:
            X_l, X_h: (B, T1, N, D)
        """
        _check_shape(X, 4, "X")
        B, T1, N, F = X.shape
        if F != self.in_feat_total:
            raise ValueError(f"Expected last dim 362 (return, trend, 360 factors), got {F}")

        # Separate channels
        ret = X[..., 0]          # (B,T1,N)
        trend = X[..., 1]        # (B,T1,N)
        factors = X[..., 2:]     # (B,T1,N,360)

        # Apply DWT over time for each stock independently
        # Shape: (B*N, 1, T1)
        ret_bn = ret.permute(0, 2, 1).contiguous().view(B * N, 1, T1)

        low_bn = self.dwt_low(ret_bn)   # (B*N,1,T_low)
        high_bn = self.dwt_high(ret_bn) # (B*N,1,T_low)

        # Inverse transform (upsample) back to a length close to T1
        low_up_bn = self.idwt_low(low_bn)   # (B*N,1,T_rec)
        high_up_bn = self.idwt_high(high_bn)

        T_rec = low_up_bn.shape[-1]
        # Center-crop or pad to exact T1
        if T_rec > T1:
            start = (T_rec - T1) // 2
            low_up_bn = low_up_bn[..., start:start+T1]
            high_up_bn = high_up_bn[..., start:start+T1]
        elif T_rec < T1:
            pad_total = T1 - T_rec
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            low_up_bn = F.pad(low_up_bn, (pad_left, pad_right))
            high_up_bn = F.pad(high_up_bn, (pad_left, pad_right))

        # Back to (B,T1,N)
        low_up = low_up_bn.view(B, N, T1).permute(0, 2, 1).contiguous()
        high_up = high_up_bn.view(B, N, T1).permute(0, 2, 1).contiguous()

        # Build low/high branch features: [low_ret/high_ret, trend, factors]
        low_cat = torch.cat([
            low_up.unsqueeze(-1),
            trend.unsqueeze(-1),
            factors,
        ], dim=-1)  # (B,T1,N,362)

        high_cat = torch.cat([
            high_up.unsqueeze(-1),
            trend.unsqueeze(-1),
            factors,
        ], dim=-1)  # (B,T1,N,362)

        X_l = self.proj_low(low_cat)   # (B,T1,N,D)
        X_h = self.proj_high(high_cat) # (B,T1,N,D)
        return X_l, X_h


############################################################
# Temporal Modules (Sec. 4.3.1, Eq. (5)-(7))
############################################################

class TemporalSelfAttention(nn.Module):
    """Temporal self-attention on low-frequency branch (Eq. (7)).

    For each stock n, apply self-attention along time:
        ta_n = Att(X_l^n, X_l^n, X_l^n)

    Input:  X_l: (B, T1, N, D)
    Output: X_tatt_l: (B, T1, N, D)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, X_l: torch.Tensor) -> torch.Tensor:
        _check_shape(X_l, 4, "X_l")
        B, T1, N, D = X_l.shape
        # (B,N,T1,D) -> (B*N,T1,D)
        X_flat = X_l.permute(0, 2, 1, 3).contiguous().view(B * N, T1, D)
        residual = X_flat
        out, _ = self.attn(X_flat, X_flat, X_flat)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = out.view(B, N, T1, D).permute(0, 2, 1, 3).contiguous()
        return out


class DilatedCausalConv(nn.Module):
    """Dilated causal convolution on high-frequency branch (Eq. (5)-(6)).

    Input:  X_h: (B, T1, N, D)
    Output: X_conv_h: (B, T1, N, D)
    """

    def __init__(self, hidden_dim: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, X_h: torch.Tensor) -> torch.Tensor:
        _check_shape(X_h, 4, "X_h")
        B, T1, N, D = X_h.shape
        X_flat = X_h.permute(0, 2, 3, 1).contiguous().view(B * N, D, T1)  # (B*N,D,T1)
        residual = X_flat
        # Causal padding on the left
        pad_len = (self.kernel_size - 1) * self.dilation
        X_padded = F.pad(X_flat, (pad_len, 0))
        out = self.conv(X_padded)  # (B*N,D,T1)
        out = F.relu(out)
        out = self.dropout(out)
        out = out + residual
        out = out.view(B, N, D, T1).permute(0, 3, 1, 2).contiguous()  # (B,T1,N,D)
        out = self.norm(out)
        return out


############################################################
# Time-Slot and Spatial Graph Embedding (Sec. 4.3.2)
############################################################

class TimeSlotEmbedding(nn.Module):
    """Time-slot embedding based on temporal graph G' (simplified).

    Paper:
        - Time slots indexed tp \in [0, 252).
        - One-hot O_i in R^{|V'|}, projected via W*_t^T to D-dim embedding.
        - W*_t initialized by temporal graph embedding.

    Implementation:
        - nn.Embedding(|V'|, D) used to approximate W*_t^T.
        - External code should map real timestamps to slot indices tp % 252.

    Input:
        time_slots: (B, T1) integer indices in [0, num_slots)
        N: number of stocks
    Output:
        rho_tem: (B, T1, N, D)
    """

    def __init__(self, hidden_dim: int, num_slots: int = 252):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_slots = num_slots
        self.embed = nn.Embedding(num_slots, hidden_dim)

    def forward(self, time_slots: torch.Tensor, N: int) -> torch.Tensor:
        _check_shape(time_slots, 2, "time_slots")
        B, T1 = time_slots.shape
        em = self.embed(time_slots)  # (B,T1,D)
        em = em.unsqueeze(2).expand(B, T1, N, self.hidden_dim)
        return em


class SpatialGraphEmbedding(nn.Module):
    """Spatial graph embedding ρ_spa for N stocks (simplified Struc2Vec).

    Paper:
        - Compute Spearman correlation N×N matrix, build adjacency graph.
        - Run Struc2Vec to get high-dimensional stock embeddings ρ_spa \in R^{N×D}.

    Implementation:
        - Use a learnable nn.Embedding(N,D) as a placeholder for ρ_spa.
        - External code can load precomputed Struc2Vec embeddings into this layer.

    Output:
        rho_spa: (B, T1, N, D)
    """

    def __init__(self, num_stocks: int, hidden_dim: int):
        super().__init__()
        self.num_stocks = num_stocks
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(num_stocks, hidden_dim)

    def forward(self, B: int, T1: int) -> torch.Tensor:
        stock_ids = torch.arange(self.num_stocks, device=self.embed.weight.device)
        em = self.embed(stock_ids).view(1, 1, self.num_stocks, self.hidden_dim)
        em = em.expand(B, T1, self.num_stocks, self.hidden_dim)
        return em


############################################################
# Graph Attention (Sec. 4.3.2, Eq. (10))
############################################################

class GraphAttentionLayer(nn.Module):
    """Graph attention layer over stocks at each time step (simplified GAT).

    Paper:
        - Uses ˜X = X + ρ_spa + ρ_tem as input.
        - Applies attention over graph nodes (stocks) at each time t.

    Implementation:
        - Uses MultiheadAttention across the node dimension per time step.
        - Does not enforce adjacency masking; external code may extend this.

    Input:
        X:       (B, T1, N, D)
        rho_spa: (B, T1, N, D)
        rho_tem: (B, T1, N, D)
    Output:
        X_gat:   (B, T1, N, D)
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, X: torch.Tensor, rho_spa: torch.Tensor, rho_tem: torch.Tensor) -> torch.Tensor:
        _check_shape(X, 4, "X")
        B, T1, N, D = X.shape
        X_tilde = X + rho_spa + rho_tem    # (B,T1,N,D)
        X_flat = X_tilde.view(B * T1, N, D)
        residual = X_flat
        out, _ = self.attn(X_flat, X_flat, X_flat)
        out = self.dropout(out)
        out = self.norm(out + residual)
        out = out.view(B, T1, N, D)
        return out


############################################################
# Dual-Frequency Spatiotemporal Encoder (Sec. 4.3)
############################################################

class DualFrequencySpatioTemporalEncoder(nn.Module):
    """Dual-Frequency Spatiotemporal Encoder.

    Components per layer (L layers total):
        - TemporalSelfAttention on low-frequency branch X_l.
        - DilatedCausalConv on high-frequency branch X_h.
        - GraphAttentionLayer on both branches using ρ_spa and ρ_tem.

    Input:
        X_l, X_h: (B, T1, N, D)
        rho_spa, rho_tem: (B, T1, N, D)
    Output:
        X_l_enc, X_h_enc: (B, T1, N, D)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_heads = int(cfg.get("num_heads", 1))
        self.num_layers = int(cfg.get("num_layers", 2))  # L in the paper
        self.kernel_size = int(cfg.get("kernel_size", 2))  # J in Eq. (5)
        self.dropout = float(cfg.get("dropout", 0.2))

        self.temp_attn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        self.graph_layers_low = nn.ModuleList()
        self.graph_layers_high = nn.ModuleList()

        for l in range(self.num_layers):
            self.temp_attn_layers.append(
                TemporalSelfAttention(self.hidden_dim, self.num_heads, self.dropout)
            )
            # Dilations 1,2,4,... approx. stacking of dilated causal convs
            dilation = 2 ** l
            self.conv_layers.append(
                DilatedCausalConv(self.hidden_dim, self.kernel_size, dilation, self.dropout)
            )
            self.graph_layers_low.append(
                GraphAttentionLayer(self.hidden_dim, self.num_heads, self.dropout)
            )
            self.graph_layers_high.append(
                GraphAttentionLayer(self.hidden_dim, self.num_heads, self.dropout)
            )

    def forward(
        self,
        X_l: torch.Tensor,
        X_h: torch.Tensor,
        rho_spa: torch.Tensor,
        rho_tem: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_shape(X_l, 4, "X_l")
        _check_shape(X_h, 4, "X_h")
        for l in range(self.num_layers):
            X_l = self.temp_attn_layers[l](X_l)
            X_h = self.conv_layers[l](X_h)
            X_l = self.graph_layers_low[l](X_l, rho_spa, rho_tem)
            X_h = self.graph_layers_high[l](X_h, rho_spa, rho_tem)
        return X_l, X_h


############################################################
# Dual-Frequency Fusion Decoder (Sec. 4.4, Eq. (11))
############################################################

class DualFrequencyFusionDecoder(nn.Module):
    """Dual-Frequency Fusion Decoder.

    Paper (Sec. 4.4.1, Eq. (11)):
        - Predictors map encoder outputs to future multi-step representations
          \hat{Y}^f_l, \hat{Y}^f_h \in R^{T2×N×D}.
        - Fusion attention:
              fa_n = Att(\hat{Y}^f_{l,n}, \hat{Y}^f_{l,n}, \hat{Y}^f_{l,n})
                    + Att(\hat{Y}^f_{l,n}, \hat{Y}^f_{h,n}, \hat{Y}^f_{h,n})
          concatenated across stocks to form \hat{Y}^f.
        - Separate FC heads for regression and classification tasks on both
          fused representation and low-frequency representation.

    Implementation:
        - Uses a 3-layer MLP predictor as in the paper's Fig. 6.
        - Applies fusion attention as per Eq. (11).

    Input:
        X_l_enc, X_h_enc: (B, T1, N, D)
    Output:
        Y_reg:  (B, T2, N)
        Y_lreg: (B, T2, N)
        P_cla:  (B, T2, N, 2)
        P_lcla:(B, T2, N, 2)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_heads = int(cfg.get("num_heads", 1))
        self.dropout = float(cfg.get("dropout", 0.2))
        self.T1 = int(cfg.get("lookback_window", 20))
        self.T2 = int(cfg.get("predict_window", 2))

        # 3-layer MLP predictor
        def make_mlp(in_dim, hidden_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        # Predictors: (B,N,D) -> (B,N,T2,D)
        self.pred_low = make_mlp(self.hidden_dim, self.hidden_dim, self.T2 * self.hidden_dim)
        self.pred_high = make_mlp(self.hidden_dim, self.hidden_dim, self.T2 * self.hidden_dim)

        # Fusion attention modules
        self.attn_self = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.attn_cross = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.dropout_layer = nn.Dropout(self.dropout)
        self.norm = nn.LayerNorm(self.hidden_dim)

        # Output heads
        self.fc_reg = nn.Linear(self.hidden_dim, 1)
        self.fc_reg_low = nn.Linear(self.hidden_dim, 1)
        self.fc_cla = nn.Linear(self.hidden_dim, 2)
        self.fc_cla_low = nn.Linear(self.hidden_dim, 2)

    def forward(
        self,
        X_l_enc: torch.Tensor,
        X_h_enc: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_shape(X_l_enc, 4, "X_l_enc")
        B, T1, N, D = X_l_enc.shape
        if T1 != self.T1:
            raise ValueError(f"Encoder output T1={T1} must match config lookback_window={self.T1}")

        # Use last time step from encoder as summary feature for prediction
        Xl_last = X_l_enc[:, -1]  # (B,N,D)
        Xh_last = X_h_enc[:, -1]  # (B,N,D)

        # Predict future representations per stock via 3-layer MLP
        # (B,N,D) -> (B,N,T2,D)
        Yl_pred_flat = self.pred_low(Xl_last).view(B, N, self.T2, D)
        Yh_pred_flat = self.pred_high(Xh_last).view(B, N, self.T2, D)

        # Rearrange to (B*N,T2,D) for attention
        Yl_flat = Yl_pred_flat.view(B * N, self.T2, D)
        Yh_flat = Yh_pred_flat.view(B * N, self.T2, D)

        # Fusion attention: self + cross (Eq. (11))
        self_res = Yl_flat
        self_out, _ = self.attn_self(Yl_flat, Yl_flat, Yl_flat)
        self_out = self.dropout_layer(self_out)
        self_out = self.norm(self_out + self_res)

        cross_res = self_out
        cross_out, _ = self.attn_cross(self_out, Yh_flat, Yh_flat)
        cross_out = self.dropout_layer(cross_out)
        fused_flat = self.norm(cross_out + cross_res)  # (B*N,T2,D)

        # Back to (B,T2,N,D)
        fused = fused_flat.view(B, N, self.T2, D).permute(0, 2, 1, 3).contiguous()
        Yl = Yl_flat.view(B, N, self.T2, D).permute(0, 2, 1, 3).contiguous()

        # Regression outputs
        Y_reg = self.fc_reg(fused).squeeze(-1)     # (B,T2,N)
        Y_lreg = self.fc_reg_low(Yl).squeeze(-1)   # (B,T2,N)

        # Classification outputs (logits)
        P_cla = self.fc_cla(fused)                 # (B,T2,N,2)
        P_lcla = self.fc_cla_low(Yl)               # (B,T2,N,2)

        return Y_reg, Y_lreg, P_cla, P_lcla


############################################################
# Full Stockformer Model (Fig. 1, Sec. 4)
############################################################

class Stockformer(nn.Module):
    """Stockformer: Price-Volume Factor Stock Selection Model.

    Architecture as in Fig. 1:
        1) Decoupling Flow Layer with DWT-based return decomposition.
        2) Dual-Frequency Spatiotemporal Encoder (temporal + graph attention).
        3) Dual-Frequency Fusion Decoder with multitask heads.

    Input:
        X:          (B, T1, N, 362)
            - channel 0: return r_t
            - channel 1: trend indicator (0/1)
            - channels 2..361: 360 Alpha360 factors
        time_slots: (B, T1) integer time-slot indices in [0,252)

    Output:
        Y_reg:  (B, T2, N)
        Y_lreg: (B, T2, N)
        P_cla:  (B, T2, N, 2)
        P_lcla:(B, T2, N, 2)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = int(cfg.get("hidden_dim", 128))
        self.num_stocks = int(cfg.get("num_stocks", 300))
        self.lookback = int(cfg.get("lookback_window", 20))
        self.predict_window = int(cfg.get("predict_window", 2))

        self.decouple = DecouplingFlowLayer(cfg)
        self.time_embed = TimeSlotEmbedding(self.hidden_dim, num_slots=252)
        self.spatial_embed = SpatialGraphEmbedding(self.num_stocks, self.hidden_dim)
        self.encoder = DualFrequencySpatioTemporalEncoder(cfg)
        self.decoder = DualFrequencyFusionDecoder(cfg)

    def forward(
        self,
        X: torch.Tensor,
        time_slots: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        _check_shape(X, 4, "X")
        _check_shape(time_slots, 2, "time_slots")
        B, T1, N, F = X.shape
        if T1 != self.lookback:
            raise ValueError(f"Input time length T1={T1} must equal lookback_window={self.lookback}")
        if N != self.num_stocks:
            raise ValueError(f"Input N={N} must equal num_stocks={self.num_stocks}")
        if F != 362:
            raise ValueError(f"Input feature dim={F} must be 362 (return, trend, 360 factors)")

        # 1) Decoupling Flow Layer
        X_l, X_h = self.decouple(X)  # (B,T1,N,D)

        # 2) Graph positional embeddings
        rho_tem = self.time_embed(time_slots, N)  # (B,T1,N,D)
        rho_spa = self.spatial_embed(B, T1)       # (B,T1,N,D)

        # 3) Dual-Frequency Spatiotemporal Encoder
        X_l_enc, X_h_enc = self.encoder(X_l, X_h, rho_spa, rho_tem)

        # 4) Dual-Frequency Fusion Decoder
        Y_reg, Y_lreg, P_cla, P_lcla = self.decoder(X_l_enc, X_h_enc)
        return Y_reg, Y_lreg, P_cla, P_lcla


############################################################
# Multi-Supervision Loss (Sec. 4.4.2, Eq. (12)-(14))
############################################################

def compute_loss(
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    targets: torch.Tensor,
    lambda_cla: float = 2.0,
) -> torch.Tensor:
    """Compute multi-task loss L = L_reg + λ L_cla.

    Paper:
        - Regression loss L_reg: MAE on returns and low-frequency component
          (Eq. (13)). 这里用 y_true 同时监督 Y_reg 和 Y_lreg。
        - Classification loss L_cla: sum of cross-entropy for main and
          low-frequency classification outputs (Eq. (14)).

    Args:
        outputs: (Y_reg, Y_lreg, P_cla, P_lcla)
            Y_reg, Y_lreg: (B,T2,N)
            P_cla, P_lcla: (B,T2,N,2) logits
        targets: (B,T2,N,2)
            targets[...,0] = true returns
            targets[...,1] = true trend labels (0 or 1)
        lambda_cla: λ balancing regression and classification terms.

    Returns:
        scalar tensor: total loss
    """
    Y_reg, Y_lreg, P_cla, P_lcla = outputs
    _check_shape(targets, 4, "targets")
    y_true = targets[..., 0]           # (B,T2,N)
    y_trend = targets[..., 1].long()   # (B,T2,N)

    # Regression loss (MAE)
    loss_reg_main = F.l1_loss(Y_reg, y_true)
    loss_reg_low = F.l1_loss(Y_lreg, y_true)
    L_reg = loss_reg_main + loss_reg_low

    # Classification loss (cross-entropy)
    B, T2, N = y_trend.shape
    P_main_flat = P_cla.view(B * T2 * N, 2)
    P_low_flat = P_lcla.view(B * T2 * N, 2)
    labels_flat = y_trend.view(B * T2 * N)

    L_cla_main = F.cross_entropy(P_main_flat, labels_flat)
    L_cla_low = F.cross_entropy(P_low_flat, labels_flat)
    L_cla = L_cla_main + L_cla_low

    total_loss = L_reg + lambda_cla * L_cla
    return total_loss
