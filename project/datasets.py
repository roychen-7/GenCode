import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH


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

        # fields 顺序: [feature1, feature2, ..., label]
        raw = handler.groupby("instrument")

        X_list, y_list = [], []

        for inst, df in raw:
            df = df.dropna()

            values = df.values  # numpy array
            num_features = len(features)

            # 2. 遍历每个股票序列
            for i in range(len(values) - lookback_window - 1):
                # 输入窗口
                x_window = values[i:i+lookback_window, :num_features]
                # label 是窗口后的第1天
                y = values[i+lookback_window, num_features]

                X_list.append(x_window)
                y_list.append(y)

        self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
        self.y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_full_dataset(
    instruments="csi300",
    start_date="2020-01-01",
    end_date="2020-12-31",
    features=["$close"],
    label="Ref($close, -1)/$close-1",
    batch_size=1024,
    shuffle=True,
    lookback_window=1,
):
    # 1. 确保 Qlib 初始化过
    qlib.init(provider_uri="./data/cn_data", region="cn")

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
