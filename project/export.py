import random
import qlib
from qlib.data import D

if __name__ == '__main__':
    # 1) 初始化 Qlib
    qlib.init(provider_uri="./data/cn_data", region="cn")

    # 2) 读取股票池文件
    print("加载股票数据...")
    instruments_file = "./data/cn_data/instruments/csi300.txt"

    with open(instruments_file, 'r') as f:
        # 读取股票代码（第一列）
        instruments_list = [line.split('\t')[0] for line in f.readlines()]

    print(f"股票池大小: {len(instruments_list)}")

    # 3) 随机挑选股票
    sample_size = min(1, len(instruments_list))
    sample_stocks = random.sample(instruments_list, sample_size)
    print(f"随机选取股票：{sample_stocks}")

    # 4) 读取原始价格数据（未归一化）
    # 使用基本的价格和成交量字段
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

    df = D.features(
        instruments=sample_stocks,
        fields=fields,
        start_time="2020-01-01",
        end_time="2020-12-31",
        freq='day'
    )

    # 5) 将 index 展开（date, instrument) 变成列
    df = df.reset_index()

    print(f"总共 {len(df)} 条记录")
    print(f"列名: {list(df.columns)}")

    # 6) 输出 CSV
    output = "alpha158_sample1.csv"
    df.to_csv(output, index=False)
    print(f"已导出到: {output}")
