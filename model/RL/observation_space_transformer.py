# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/13 10:14
# Contact       : hucheng.lew@qq.com
# File          : observation_space_transformer.py
# IDE           : PyCharm
# Description   : 观察空间转换器


from feature_normalize import ContinuesRatioNormalizer
from typing import Tuple
from gemm_space import GEMMObservationSpace



class ObservationSpaceTransformer:
    """
    观察空间转换器
    """
    def __init__(self, dim_m: Tuple[int, int], dim_k: Tuple[int, int], dim_n: Tuple[int, int]):
        """
        构造函数
        :param dim_m: M维度信息 (min, max)
        :param dim_k: K维度信息 (min, max)
        :param dim_n: N维度信息 (min, max)
        :param space_min: 归一化空间最小值
        :param space_max: 归一化空间最大值
        """
        m_space_min, m_space_max = GEMMObservationSpace.get_m_range()
        k_space_min, k_space_max = GEMMObservationSpace.get_k_range()
        n_space_min, n_space_max = GEMMObservationSpace.get_n_range()
        self._dim_m_normalizer = ContinuesRatioNormalizer(input_min=dim_m[0], input_max=dim_m[1],
                                                           output_min=m_space_min, output_max=m_space_max)
        self._dim_k_normalizer = ContinuesRatioNormalizer(input_min=dim_k[0], input_max=dim_k[1],
                                                           output_min=k_space_min, output_max=k_space_max)
        self._dim_n_normalizer = ContinuesRatioNormalizer(input_min=dim_n[0], input_max=dim_n[1],
                                                           output_min=n_space_min, output_max=n_space_max)

    def normalize(self, m: int, k: int, n: int) -> Tuple[float, float, float]:
        """
        归一化操作
        :param m: M维度特征值
        :param k: K维度特征值
        :param n: N维度特征值
        :return: 归一化后的 (M, K, N) 特征值
        """
        m_norm = self._dim_m_normalizer.normalize(m)
        k_norm = self._dim_k_normalizer.normalize(k)
        n_norm = self._dim_n_normalizer.normalize(n)
        return m_norm, k_norm, n_norm


    def denormalize(self, m_norm: float, k_norm: float, n_norm: float) -> Tuple[int, int, int]:
        """
        反归一化操作
        :param m_norm: M维度归一化特征值
        :param k_norm: K维度归一化特征值
        :param n_norm: N维度归一化特征值
        :return: 反归一化后的 (M, K, N) 特征值
        """
        m_value = self._dim_m_normalizer.denormalize(m_norm)
        k_value = self._dim_k_normalizer.denormalize(k_norm)
        n_value = self._dim_n_normalizer.denormalize(n_norm)
        return m_value, k_value, n_value
