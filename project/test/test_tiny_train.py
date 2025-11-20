import json
import os
import subprocess
import sys

def test_tiny_train_runs():
    if os.path.exists("tiny_train_log.json"):
        os.remove("tiny_train_log.json")

    result = subprocess.run(
        [sys.executable, "train.py", "--mode", "test"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=40,
    )

    # 确保脚本没有报错
    assert result.returncode == 0

    # 确保 tiny mode 的 log 文件存在
    assert os.path.exists("tiny_train_log.json")

    # 检查 loss 变化
    data = json.load(open("tiny_train_log.json"))
    losses = data["loss_history"]

    assert losses[0] > 0
    assert losses[-1] <= losses[0] * 1.5  # 允许轻微浮动
