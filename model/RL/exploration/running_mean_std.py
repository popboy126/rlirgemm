# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/23 15:57
# Contact       : hucheng.lew@qq.com
# File          : running_mean_std.py
# IDE           : PyCharm
# Description   : 在线归一化

import torch
from typing import Tuple
import numpy as np


class RunningMeanStd:
    """在线计算数据流的均值与标准差，使用Welford‘s online algorithm。"""
    def __init__(self, shape: Tuple[int], epsilon:float =1e-4):
        """
        初始化
        """
        self._shape = shape
        self._epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon # 初始化为一个极小值，防止除零

    def update(self, x: np.ndarray):
        """批量更新统计量。x是一个numpy数组或PyTorch张量。"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int):
        # 根据新旧批次合并统计量
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = M2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x, clip_range: Tuple[float, float]=(-1.0, 1.0)):
        """归一化x，并可选进行裁剪。"""
        std = np.sqrt(self.var + 1e-8)
        std = np.maximum(std, 1e-4)  # ★ 核心防御：标准差绝对不能低于 1e-4
        x_norm = (x - self.mean) / std
        if clip_range is not None:
            x_norm = np.clip(x_norm, clip_range[0], clip_range[1])
        return x_norm

    def reset(self):
        """
        重置RunningMeanStd
        """
        self.mean = np.zeros(self._shape, dtype=np.float64)
        self.var = np.ones(self._shape, dtype=np.float64)
        self.count = self._epsilon # 初始化为一个极小值，防止除零

