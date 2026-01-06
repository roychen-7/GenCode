import torch
from torch.utils.data import Dataset
import numpy as np
from qlib.data import D

class QlibDataset(Dataset):
    def __init__(self, instruments, start_date, end_date,
                 features, label, lookback_window):

        # 1. 创建 Qlib 数据 handler
        # 注意：必须把未来 2 天的收益都拉出来
        #   day+1: Ref($close, -1)/$close - 1
        #   day+2: Ref($close, -2)/$close - 1
        label_day1 = "Ref($close, -1)/$close - 1"
        label_day2 = "Ref($close, -2)/$close - 1"

        handler = D.features(
            instruments=D.instruments(instruments),
            fields=features + [label_day1, label_day2],
            start_time=start_date,
            end_time=end_date,
        )

        raw = handler.groupby("instrument")

        X_list, y1_list, y2_list = [], [], []

        for inst, df in raw:
            df = df.dropna()
            values = df.values  # numpy array
            num_features = len(features)

            # features: 0 .. num_features-1
            # y1      : num_features
            # y2      : num_features+1

            for i in range(len(values) - lookback_window - 2):
                # ----------------------
                # X 过去 lookback_window 天
                # ----------------------
                x_window = values[i:i+lookback_window, :num_features]

                # ----------------------
                # y1 = 未来第1天收益
                # y2 = 未来第2天收益
                # ----------------------
                y1 = values[i + lookback_window, num_features]
                y2 = values[i + lookback_window + 1, num_features + 1]

                X_list.append(x_window)
                y1_list.append(y1)
                y2_list.append(y2)

        # 转 numpy
        if len(X_list) == 0:
            raise ValueError("No data samples found. Check your date range and instruments.")

        X = np.array(X_list, dtype=np.float32)          # (N, T1, F)
        y1 = np.array(y1_list, dtype=np.float32)        # (N,)
        y2 = np.array(y2_list, dtype=np.float32)        # (N,)

        # ----------------------
        # 全局标准化 X
        # ----------------------
        X_2d = X.reshape(-1, X.shape[-1])
        mean = X_2d.mean(axis=0, keepdims=True)
        std = X_2d.std(axis=0, keepdims=True) + 1e-8
        X_norm = ((X_2d - mean) / std).reshape(X.shape)

        # reshape → (B, T1, 1, F)
        X_norm_4d = X_norm[:, :, None, :]

        # pad 到 362 维
        F = X_norm_4d.shape[-1]
        if F < 362:
            padding = np.zeros((X_norm_4d.shape[0], X_norm_4d.shape[1], 1, 362 - F), dtype=np.float32)
            X_norm_4d = np.concatenate([X_norm_4d, padding], axis=-1)

        self.X = torch.tensor(X_norm_4d, dtype=torch.float32)

        # ----------------------
        # ⭐构建 Stockformer 需要的 Y: (B, T2=2, N=1, 2)
        # ----------------------
        # (N,2) → returns for day1/day2
        returns_2day = np.stack([y1, y2], axis=1)  # (N,2)

        # 趋势标签（0/1）
        trends_2day = (returns_2day >= 0).astype(np.int64)  # (N,2)

        # reshape → (N, T2=2, 1, 2)
        y_4d = np.stack([returns_2day, trends_2day], axis=-1)  # (N,2,2)
        y_4d = y_4d.reshape(len(X), 2, 1, 2)

        self.y = torch.tensor(y_4d, dtype=torch.float32)

        # 简单时间槽
        self.time_slots = torch.arange(lookback_window).unsqueeze(0).repeat(len(self.X), 1)

        # 保存 mean/std
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.time_slots[idx]
