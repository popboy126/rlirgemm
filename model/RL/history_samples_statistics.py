# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/12 16:14
# Contact       : hucheng.lew@qq.com
# File          : history_samples_statistics.py
# IDE           : PyCharm
# Description   : 历史样本统计

from typing import Union, List, Tuple, Optional
import itertools
import logging
from exploration.cache_manager import DiscreteActionCacheManager
from multiprocessing.synchronize import Lock


class HistorySampleStatistics:
    """
    历史样本信息统计
    """
    def __init__(self,
                 name: str,
                 loop_orders: List[int],
                 packing_types: List[int],
                 thread_types: List[List[int]],
                 shm_name: str = None,
                 lock: Lock = None,
                 element_byte_size: int = 4,
                 logger: Union[None, logging.Logger] = None
                 ):
        """
        初始化
        """
        self._name = name
        threads = [tuple(k) for k in thread_types]

        self._disc_action_stat = DiscreteActionCacheManager(
            loop_orders=loop_orders,
            packing_types=packing_types,
            thread_types=threads,
            shm_name=shm_name,
            create=False,
            lock=lock,
        )

        self._element_byte_size = element_byte_size

        self._total_count = 0
        self._total_invalid_count = 0
        self._logger = logger

    def __del__(self):
        """
        析构函数
        """
        self._disc_action_stat.release()

    def add_sample(self, loop_order: int, packing_type: int, thread_type: Tuple[int, int, int],  reward: float, performance: Optional[float] = None) -> None:
        """
        添加样本
        """
        if performance is not None and performance > 0:
            is_valid = True
        else:
            is_valid = False

        log_str = self._disc_action_stat.update(
            loop_order=loop_order,
            packing_type=packing_type,
            thread_type=thread_type,
            is_valid=is_valid,
        )

        if log_str:
            self._logger.info(log_str)


    def get_discrete_action_ratio(self, loop_order: int, packing_type: int, thread_type: Tuple[int, int, int]) -> Tuple[float, float]:
        """
        获得指定离散动作的概率
        """
        return self._disc_action_stat.get_discrete_action_ratio(loop_order, packing_type, thread_type)
