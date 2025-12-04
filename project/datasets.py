import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from qlib.data import D

class QlibDataset(Dataset):
    def __init__(self, instruments, start_date, end_date,
                 features, label, lookback_window):

        # 1. 创建 Qlib 数据 handler
        handler = D.features(
            instruments=D.instruments(instruments),
            fields=features + [label],  # 一起拉取
            start_time=start_date,
            end_time=end_date,
        )

        raw = handler.groupby("instrument")

        X_list, y_list = [], []

        for inst, df in raw:
            df = df.dropna()
            values = df.values  # numpy array
            num_features = len(features)

            # ----------------------
            # 先收集所有 X_window 原始特征
            # ----------------------
            for i in range(len(values) - lookback_window):
                x_window = values[i:i+lookback_window, :num_features]
                y = values[i+lookback_window-1, num_features]
                X_list.append(x_window)
                y_list.append(y)

        # ----------------------
        # 转成 numpy
        # X shape: (N, T, F)
        # y shape: (N,)
        # ----------------------
        if len(X_list) == 0:
            raise ValueError("No data samples found. Check your date range and instruments.")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.float32).reshape(-1, 1)

        # ----------------------
        # ⭐ 重点：对 X 做标准化 (全局 mean/std)
        # 计算方式：对所有样本的所有时间步的每个 feature 计算 mean/std
        # ----------------------
        # X reshape: (N*T, F)
        X_2d = X.reshape(-1, X.shape[-1])

        mean = X_2d.mean(axis=0, keepdims=True)            # (1, F)
        std = X_2d.std(axis=0, keepdims=True) + 1e-8       # (1, F)

        # 标准化
        X_norm = (X_2d - mean) / std

        # reshape 回 (N, T, F)
        X_norm = X_norm.reshape(X.shape)

        # ----------------------
        # 转成 tensor 作为最终数据
        # Reshape to (N, T, F) -> (B, T, 1, F) where we treat each sample as a single stock
        # ----------------------
        X_norm_4d = X_norm[:, :, None, :]  # (N, T, 1, F)

        # Pad features to 362 dimensions
        F = X_norm_4d.shape[-1]
        if F < 362:
            padding = np.zeros((X_norm_4d.shape[0], X_norm_4d.shape[1], 1, 362 - F), dtype=np.float32)
            X_norm_4d = np.concatenate([X_norm_4d, padding], axis=-1)

        self.X = torch.tensor(X_norm_4d, dtype=torch.float32)

        # Y needs to be (B, T2, N, 2) - we'll use T2=2 and add trend as 0
        # Duplicate the y value for both prediction windows
        y_4d = np.repeat(y[:, None, None, :], 2, axis=1)  # (N, 2, 1, 1)
        trend = np.zeros_like(y_4d)
        y_4d = np.concatenate([y_4d, trend], axis=-1)  # (N, 2, 1, 2)
        self.y = torch.tensor(y_4d, dtype=torch.float32)

        # Create time_slots (simple sequential indices)
        self.time_slots = torch.arange(lookback_window).unsqueeze(0).repeat(len(self.X), 1)

        # 可选：保存 mean/std 用于预测还原
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.time_slots[idx]

