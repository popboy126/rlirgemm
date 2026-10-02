# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/2/2 13:58
# Contact       : hucheng.lew@qq.com
# File          : gemm_adam_optimizer.py
# IDE           : PyCharm
# Description   : 自定义adam优化器

import torch
from torch.optim import Adam
from torch.nn.parameter import Parameter
from tianshou.algorithm.optim import OptimizerFactory
from torch.optim.lr_scheduler import LRScheduler
from typing import Tuple
from collections.abc import Iterator

class GemmAdamOptimizer(OptimizerFactory):
    """
    自定义的adam优化器，使用GEMM加速计算
    """
    def __init__(self, base_lr: float,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__()
        self.base_lr = base_lr
        self.attention_lr = base_lr * 0.5
        self.actor_head_lr = base_lr * 1.0
        self.critic_head_lr = base_lr * 1.5
        self.weight_decay = weight_decay
        self.eps = eps
        self.betas = betas


    def create_instances(
            self,
            module: torch.nn.Module,
    ) -> tuple[torch.optim.Optimizer, LRScheduler | None]:
        optimizer = self._create_optimizer_for_params(module.named_parameters())
        lr_scheduler = None
        if self.lr_scheduler_factory is not None:
            lr_scheduler = self.lr_scheduler_factory.create_scheduler(optimizer)
        return optimizer, lr_scheduler

    def _create_optimizer_for_params(self, params: Iterator[tuple[str, Parameter]]) -> torch.optim.Optimizer:
        """
            创建GEMM加速的Adam优化器
        """
        # 1. 定义不同的参数组列表
        optimizer_grouped_parameters = []
        for name, param in params:
            if "_attention" in name:
                optimizer_grouped_parameters.append({
                    'params': param,
                    'lr': self.attention_lr,  # attention部分学习率减半
                })
            elif "_head" in name:
                optimizer_grouped_parameters.append({
                    'params': param,
                    'lr': self.actor_head_lr,  # head部分学习率设置为基准的两倍
                })
            elif "_continuous_net" in name:
                optimizer_grouped_parameters.append({
                    'params': param,
                    'lr': self.actor_head_lr,  # prfm_stride_pre_prepocess_net部分学习率设置为基准的0.1倍
                })
            elif "last" in name:
                optimizer_grouped_parameters.append({
                    'params': param,
                    'lr': self.critic_head_lr,  # last部分学习率设置为基准的5倍
                })
            else:
                optimizer_grouped_parameters.append({
                    'params': param,
                    'lr': self.base_lr,  # last部分学习率设置为基准的5倍
                })
        return Adam(
            optimizer_grouped_parameters,
            lr=self.base_lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay,
        )
