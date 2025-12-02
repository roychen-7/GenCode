import os
import random
import qlib
from qlib.data import D
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data", "cn_data")
    qlib.init(provider_uri=data_path, region="cn")
    feature_list = Alpha158().get_feature_config()
    # print(feature_list)

    instruments_file = "./data/cn_data/instruments/csi300.txt"

    with open(instruments_file, 'r') as f:
        # 读取股票代码（第一列）
        instruments_list = [line.split('\t')[0] for line in f.readlines()]

    # print(f"股票池大小: {len(instruments_list)}")

    # 3) 随机挑选股票
    sample_size = min(1, len(instruments_list))
    sample_stocks = random.sample(instruments_list, sample_size)
    # print(f"随机选取股票：{sample_stocks}")


    fields = [
        "$open",      # 开盘价
        "$high",      # 最高价
        "$low",       # 最低价
        "$close",     # 收盘价
        "$volume",    # 成交量
        "$vwap",      # 成交量加权平均价
        "$change",    # 涨跌额
        "$factor",    # 复权因子
    ]

    # print(fields)


    handler = Alpha158(
        instruments=["SH600000"],
        start_time="2020-01-01",
        end_time="2020-12-31",
        fit_start_time="2020-01-01",
        fit_end_time="2020-06-30"
    )

    # dataset = DatasetH(handler)
    dataset = DatasetH(handler, segments={"train": ("2020-01-01", "2020-12-31")})

    df = dataset.prepare("train")

    print("DataFrame shape:", df.shape)
    print("Column index info:")
    print("  nlevels:", df.columns.nlevels)
    print("  columns:", df.columns.tolist()[:10])  # Show first 10 columns

    if df.columns.nlevels > 1:
        real_fields = df.columns.get_level_values(1).unique().tolist()
    else:
        real_fields = df.columns.tolist()

    print("字段数量：", len(real_fields))
    for f in real_fields:  # Show first 20 fields
        print(f)


    # 6) 输出 CSV
    output = "alpha158_sample1.csv"
    df.to_csv(output, index=False)
    print(f"已导出到: {output}")

if __name__ == "__main__":
    main()
