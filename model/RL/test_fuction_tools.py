# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/1 23:22
# Contact       : hucheng.lew@qq.com
# File          : test_fuction_tools.py
# IDE           : PyCharm
# Description   :

import unittest
from math import ceil

def check_thread_dim_blocking(dim_size: int, thread_num: int, block_size: int, must_same_loop_times: bool) -> bool:
    """
    检查单个维度的线程数和分块大小是否合理
    """
    dim_per_thread = ceil(dim_size / thread_num)
    spliter_idx_set = {i for i in range(0, dim_size, dim_per_thread)}
    spliter_idx_set.add(dim_size)
    spliter_idx_list = sorted(list(spliter_idx_set))

    if len(spliter_idx_list) <= thread_num:  # 维度数比线程数少
        return False

    task_num_list = [spliter_idx_list[i + 1] - spliter_idx_list[i] for i in range(0, len(spliter_idx_list) - 1, 1)]
    dim_per_thread_min = min(task_num_list)
    dim_per_thread_max = max(task_num_list)

    if dim_per_thread_min < block_size:
        return False

    if must_same_loop_times:
        loop_times_min = ceil(dim_per_thread_min / block_size)
        loop_times_max = ceil(dim_per_thread_max / block_size)
        if loop_times_min != loop_times_max:
            return False

    return True


class TestFunctionTools(unittest.TestCase):
    def test_check_thread_dim_blocking(self):
        # 测试用例1: 合理的线程数和分块大小
        self.assertTrue(check_thread_dim_blocking(4, 1, 4, True))
