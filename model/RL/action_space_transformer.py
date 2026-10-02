# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/13 10:28
# Contact       : hucheng.lew@qq.com
# File          : action_space_transformer.py
# IDE           : PyCharm
# Description   : 动作空间转换器（方案1：在环境层做sigmoid转换）

import torch
from feature_normalize import ContinuesBucketNormalizer, ContinuesRatioNormalizer
from gemm_space import GEMMActionSpace
from typing import Tuple, List, Callable



class ActionSpaceTransformer:
    """
    动作空间转换器
    方案1：网络输出logit，在反归一化时做sigmoid转换
    """
    def __init__(self,
                 loop_orders: List[int],
                 packing_types: List[int],
                 thread_types: List[Tuple[int, int, int]],
                 action_space: GEMMActionSpace,
                 matrix_element_size: int = 4
                 ):
        """
        构造函数
        """
        self._loop_orders = loop_orders
        self._packing_types = packing_types
        self._thread_types = thread_types  # 每个线程类型是一个三元组
        self._matrix_element_size = matrix_element_size  # 矩阵元素大小，单位字节，默认4字节（float32）
        self._action_space = action_space


    def denormalize(self, m: int, k: int, n: int, mc_norm: float, kc_norm: float,
                    nc_norm: float, prfm_a_skip_bytes_norm: float,
                    prfm_b_skip_bytes_norm: float,
                    prfm_c_skip_bytes_norm: float,
                    prfm_add_c_skip_bytes_norm: float,
                    prfm_add_partial_sum_skip_bytes_norm: float) -> Tuple[int, int, int, int, int, int, int, int]:
        """
        反归一化操作
        【方案1】网络输出的是logit，需要先sigmoid转换为(0,1)范围，再计算实际物理量
        """
        # 【方案1修改】首先将logit转换为(0,1)比例
        mc_logit_to_prob = torch.sigmoid(torch.as_tensor(mc_norm))
        kc_logit_to_prob = torch.sigmoid(torch.as_tensor(kc_norm))
        nc_logit_to_prob = torch.sigmoid(torch.as_tensor(nc_norm))
        prfm_a_logit_to_prob = torch.sigmoid(torch.as_tensor(prfm_a_skip_bytes_norm))
        prfm_b_logit_to_prob = torch.sigmoid(torch.as_tensor(prfm_b_skip_bytes_norm))
        prfm_c_logit_to_prob = torch.sigmoid(torch.as_tensor(prfm_c_skip_bytes_norm))
        prfm_add_c_logit_to_prob = torch.sigmoid(torch.as_tensor(prfm_add_c_skip_bytes_norm))
        prfm_add_partial_sum_logit_to_prob = torch.sigmoid(torch.as_tensor(prfm_add_partial_sum_skip_bytes_norm))

        # 转换为标量
        mc_prob = mc_logit_to_prob.item()
        kc_prob = kc_logit_to_prob.item()
        nc_prob = nc_logit_to_prob.item()
        prfm_a_prob = prfm_a_logit_to_prob.item()
        prfm_b_prob = prfm_b_logit_to_prob.item()
        prfm_c_prob = prfm_c_logit_to_prob.item()
        prfm_add_c_prob = prfm_add_c_logit_to_prob.item()
        prfm_add_partial_sum_prob = prfm_add_partial_sum_logit_to_prob.item()

        # 1. 构建反归一化器（将(0,1)比例转换为实际物理量）
        mc_range = self._action_space.mc_range
        mc_normalizer = ContinuesRatioNormalizer(input_min=1, input_max=m,
                                                 output_min=mc_range[0], output_max=mc_range[1])
        kc_range = self._action_space.kc_range
        kc_normalizer = ContinuesRatioNormalizer(input_min=1, input_max=k,
                                                 output_min=kc_range[0], output_max=kc_range[1])
        nc_range = self._action_space.nc_range
        nc_normalizer = ContinuesRatioNormalizer(input_min=1, input_max=n,
                                                 output_min=nc_range[0], output_max=nc_range[1])
        prfm_a_skip_bytes_range = self._action_space.prfm_a_skip_bytes_range
        prfm_a_skip_bytes_normalizer = ContinuesRatioNormalizer(input_min=0, input_max=m * k * self._matrix_element_size,
                                                                output_min=prfm_a_skip_bytes_range[0], output_max=prfm_a_skip_bytes_range[1])
        prfm_b_skip_bytes_range = self._action_space.prfm_b_skip_bytes_range
        prfm_b_skip_bytes_normalizer = ContinuesRatioNormalizer(input_min=0, input_max=k * n * self._matrix_element_size,
                                                                output_min=prfm_b_skip_bytes_range[0], output_max=prfm_b_skip_bytes_range[1])
        mat_c_byte_size = m * n * self._matrix_element_size
        prfm_c_skip_bytes_range = self._action_space.prfm_c_skip_bytes_range
        prfm_c_skip_bytes_normalizer = ContinuesRatioNormalizer(input_min=0, input_max=mat_c_byte_size,
                                                                output_min=prfm_c_skip_bytes_range[0], output_max=prfm_c_skip_bytes_range[1])
        prfm_add_c_skip_bytes_range = self._action_space.prfm_add_c_skip_bytes_range
        prfm_add_c_skip_bytes_normalizer = ContinuesRatioNormalizer(input_min=0, input_max=mat_c_byte_size,
                                                                    output_min=prfm_add_c_skip_bytes_range[0], output_max=prfm_add_c_skip_bytes_range[1])
        prfm_add_partial_sum_skip_bytes_range = self._action_space.prfm_add_partial_sum_skip_bytes_range
        prfm_add_partial_sum_skip_bytes_normalizer = ContinuesRatioNormalizer(input_min=0, input_max=mat_c_byte_size,
                                                                              output_min=prfm_add_partial_sum_skip_bytes_range[0], output_max=prfm_add_partial_sum_skip_bytes_range[1])

        # 【方案1修改】使用sigmoid后的概率值进行反归一化
        mc = mc_normalizer.denormalize(mc_prob)
        kc = kc_normalizer.denormalize(kc_prob)
        nc = nc_normalizer.denormalize(nc_prob)
        prfm_a_skip_bytes = prfm_a_skip_bytes_normalizer.denormalize(prfm_a_prob)
        prfm_b_skip_bytes = prfm_b_skip_bytes_normalizer.denormalize(prfm_b_prob)
        prfm_c_skip_bytes = prfm_c_skip_bytes_normalizer.denormalize(prfm_c_prob)
        prfm_add_c_skip_bytes = prfm_add_c_skip_bytes_normalizer.denormalize(prfm_add_c_prob)
        prfm_add_partial_sum_skip_bytes = prfm_add_partial_sum_skip_bytes_normalizer.denormalize(prfm_add_partial_sum_prob)

        return mc, kc, nc, prfm_a_skip_bytes, prfm_b_skip_bytes, prfm_c_skip_bytes, prfm_add_c_skip_bytes, prfm_add_partial_sum_skip_bytes
