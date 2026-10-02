# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/4/8 08:38
# Contact       : hucheng.lew@qq.com
# File          : parse_reward_history.py
# IDE           : PyCharm
# Description   : 解析训练的历史日志

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import re
from datetime import datetime


def get_shape_actions(log_files: List[str]) -> pd.DataFrame:
    """
    获取目标形状的动作
    """
    cols = ["filename", "log_time", "m", "k", "n", "loop_order", "packing_type",
            "tm", "tk", "tn", "perf", "ema_perf", "best_perf", "perf_ratio", "best_global_avg",
            "exploration_reward", "absolute_reward", "smooth_factor", "breakthrough_reward", "improvement_pct", "negative_penalty", "total_reward"]
    result_set = set()
    REGEX = re.compile(r"""\[(\S+) (\S+),\d+ DEBUG reward_manager\.py:511]\s+\[Reward] shape=\((\d+), (\d+), (\d+)\), loop_order=(\d+), packing_type=(\d+), thread_type=\((\d+), (\d+), (\d+)\), perf=(\S+), ema=(\S+), best=(\S+), perf_ratio=(\S+), best_global_avg=(\S+), exploration_reward=(\S+), absolute_reward=(\S+) \(smooth_factor=(\S+)\), breakthrough_reward=(\S+), improvement_pct=(\S+), negative_penalty=(\S+), total_reward=(\S+)""")
    for log_file in log_files:
        basename = os.path.basename(log_file)
        if "test" in basename:
            continue

        with open(log_file, "r") as f:
            for line in f:
                match_result = REGEX.search(line)
                if match_result:
                    log_time = datetime.strptime(f"{match_result.group(1)} {match_result.group(2)}", "%Y-%m-%d %H:%M:%S")
                    m, k, n = int(match_result.group(3)), int(match_result.group(4)), int(match_result.group(5))
                    loop_order = int(match_result.group(6))
                    packing_type = int(match_result.group(7))
                    tm = int(match_result.group(8))
                    tk = int(match_result.group(9))
                    tn = int(match_result.group(10))
                    perf = float(match_result.group(11))
                    ema_perf = float(match_result.group(12))
                    best_perf = float(match_result.group(13))
                    perf_ratio = float(match_result.group(14))
                    best_global_avg = float(match_result.group(15))
                    exploration_reward = float(match_result.group(16))
                    absolute_reward = float(match_result.group(17))
                    smooth_factor = float(match_result.group(18))
                    breakthrough_reward = float(match_result.group(19))
                    improvement_pct = float(match_result.group(20))
                    negative_penalty = float(match_result.group(21))
                    total_reward = float(match_result.group(22))
                    result_set.add((
                        basename,
                        log_time, m, k, n, loop_order, packing_type,
                        tm, tk, tn, perf, ema_perf, best_perf, perf_ratio, best_global_avg,
                        exploration_reward, absolute_reward, smooth_factor, breakthrough_reward,
                        improvement_pct, negative_penalty, total_reward
                    ))

    shape_action_pd = pd.DataFrame(list(result_set), columns=cols)
    shape_action_pd.sort_values(by=["log_time", "m", "k", "n", "best_perf"], inplace=True)
    return shape_action_pd


def get_shape_perf(df: pd.DataFrame) -> pd.DataFrame:
    """
    获取形状的最大性能
    """
    result = df.groupby(by=['m', 'k', 'n'])['perf'].agg(['max', 'mean', 'median', 'min'])
    # 如果希望将类别索引重新变成一列（可选）
    result = result.reset_index()

    return result




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse RL training reward logs")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_file_path = [str(path) for path in args.log_dir.iterdir() if path.is_file()]

    shape_actions = get_shape_actions(log_file_path)
    shape_perf = get_shape_perf(shape_actions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    xlsx_output = pd.ExcelWriter(args.output)

    shape_actions.to_excel(xlsx_output, sheet_name="shape_actions")
    shape_perf.to_excel(xlsx_output, sheet_name="shape_perf")
    xlsx_output.close()

