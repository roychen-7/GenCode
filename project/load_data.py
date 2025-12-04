from project.datasets import QlibDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import qlib

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
