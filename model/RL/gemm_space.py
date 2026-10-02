# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/14 16:26
# Contact       : hucheng.lew@qq.com
# File          : gemm_action_space.py
# IDE           : PyCharm
# Description   : gemm动作空间

from typing import Tuple, List
import numpy as np
import gymnasium as gym


class GEMMObservationSpace:
    """
    定义GEMM观测空间
    """
    @classmethod
    def get_gym_space(cls) -> gym.spaces.Box:
        """
        获取空间描述
        """
        space_min = np.array([
            cls.get_m_range()[0],
            cls.get_k_range()[0],
            cls.get_n_range()[0],
        ], dtype=np.float64)

        space_max = np.array([
            cls.get_m_range()[1],
            cls.get_k_range()[1],
            cls.get_n_range()[1],
        ], dtype=np.float64)
        return gym.spaces.Box(low=space_min, high=space_max, shape=space_max.shape, dtype=np.float64)

    @classmethod
    def get_m_range(cls) -> Tuple[int, int]:
        return 0, 1

    @classmethod
    def get_k_range(cls) -> Tuple[int, int]:
        return 0, 1

    @classmethod
    def get_n_range(cls) -> Tuple[int, int]:
        return 0, 1


class GEMMActionSpace:
    """
    定义GEMM动作空间
    """
    def __init__(self, loop_order_type: List[int], packing_type: List[int], thread_type: List[Tuple[int, int, int]]):
        self.loop_order_type = loop_order_type
        self.packing_type = packing_type
        self.thread_type = thread_type

        self._loop_order_range = (0, len(loop_order_type))
        self._packing_type_range = (0, len(packing_type))
        self._thread_type_range = (0, len(thread_type))
        self._mc_range = (0, 1)
        self._kc_range = (0, 1)
        self._nc_range = (0, 1)
        self._prfm_a_skip_bytes_range = (0, 1)
        self._prfm_b_skip_bytes_range = (0, 1)
        self._prfm_c_skip_bytes_range = (0, 1)
        self._prfm_add_c_skip_bytes_range = (0, 1)
        self._prfm_add_partial_sum_skip_bytes_range = (0, 1)

    def get_gym_space(self) -> gym.spaces.Box:
        """
        获取空间描述
        """
        space_min = np.array([
            self._loop_order_range[0],
            self._thread_type_range[0],
            self._packing_type_range[0],
            self._mc_range[0],
            self._kc_range[0],
            self._nc_range[0],
            self._prfm_a_skip_bytes_range[0],
            self._prfm_b_skip_bytes_range[0],
            self._prfm_c_skip_bytes_range[0],
            self._prfm_add_c_skip_bytes_range[0],
            self._prfm_add_partial_sum_skip_bytes_range[0],
        ], dtype=np.float64)
        space_max = np.array([
            self._loop_order_range[1],
            self._thread_type_range[1],
            self._packing_type_range[1],
            self._mc_range[1],
            self._kc_range[1],
            self._nc_range[1],
            self._prfm_a_skip_bytes_range[1],
            self._prfm_b_skip_bytes_range[1],
            self._prfm_c_skip_bytes_range[1],
            self._prfm_add_c_skip_bytes_range[1],
            self._prfm_add_partial_sum_skip_bytes_range[1]
        ], dtype=np.float64)
        return gym.spaces.Box(low=space_min, high=space_max, shape=space_max.shape, dtype=np.float64)


    @property
    def loop_order_range(self) -> Tuple[int, int]:
        return self._loop_order_range

    @property
    def packing_type_range(self) -> Tuple[int, int]:
        return self._packing_type_range

    @property
    def thread_type_range(self) -> Tuple[int, int]:
        return self._thread_type_range

    @property
    def mc_range(self) -> Tuple[int, int]:
        return self._mc_range

    @property
    def kc_range(self) -> Tuple[int, int]:
        return self._kc_range

    @property
    def nc_range(self) -> Tuple[int, int]:
        return self._nc_range

    @property
    def prfm_a_skip_bytes_range(self) -> Tuple[int, int]:
        return self._prfm_a_skip_bytes_range

    @property
    def prfm_b_skip_bytes_range(self) -> Tuple[int, int]:
        return self._prfm_b_skip_bytes_range

    @property
    def prfm_c_skip_bytes_range(self) -> Tuple[int, int]:
        return self._prfm_c_skip_bytes_range

    @property
    def prfm_add_c_skip_bytes_range(self) -> Tuple[int, int]:
        return self._prfm_add_c_skip_bytes_range

    @property
    def prfm_add_partial_sum_skip_bytes_range(self) -> Tuple[int, int]:
        return self._prfm_add_partial_sum_skip_bytes_range
