import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import qlib
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
            for i in range(len(values) - lookback_window - 1):
                x_window = values[i:i+lookback_window, :num_features]
                y = values[i+lookback_window, num_features]
                X_list.append(x_window)
                y_list.append(y)

        # ----------------------
        # 转成 numpy
        # X shape: (N, T, F)
        # y shape: (N,)
        # ----------------------
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
        # ----------------------
        self.X = torch.tensor(X_norm, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        # 可选：保存 mean/std 用于预测还原
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_full_dataset(
    instruments="csi300",
    start_date="2020-01-01",
    end_date="2020-12-31",
    features=["$open", "$high", "$low", "$close", "$volume", "$factor"],
    label="Ref($close, -1)/$close-1",
    batch_size=1024,
    shuffle=True,
    lookback_window=1,
):
    # 1. 确保 Qlib 初始化过
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cn_data")
    qlib.init(provider_uri=data_path, region="cn")

    # 2. 生成 PyTorch dataset
    ds = QlibDataset(
        instruments=instruments,
        start_date=start_date,
        end_date=end_date,
        features=features,
        label=label,
        lookback_window=lookback_window,
    )

    # 3. 返回 DataLoader
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
