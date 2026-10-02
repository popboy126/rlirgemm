# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/1/23 19:04
# Contact       : hucheng.lew.qq.com
# File          : history_performance.py
# IDE           : PyCharm
# Description   : 记录历史性能（EMA基线 + 历史最佳），支持按线程配置区分

import logging
from typing import Optional, Tuple, List, Dict
from .cache_manager import ShapePerformanceCacheManager
from multiprocessing.synchronize import Lock


class HistoryBestPerformance:
    """
    记录历史性能，使用EMA（指数移动平均）作为baseline，同时保留历史最佳记录。
    解决了原来只记录最佳性能导致的"棘轮效应"：baseline只升不降，
    使得agent越训练reward越低。

    关键改进：支持按线程配置区分性能记录，避免不同线程配置的性能数据混淆。
    """

    def __init__(
        self,
        ema_alpha:float,
        shape_list: List[Tuple[int, int, int]]= None,
        shm_name: str = None,
        lock: Lock = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        初始化
        :param ema_alpha: EMA平滑系数。alpha越大，新样本权重越高，baseline跟踪越快。
                          推荐范围 [0.2, 0.4]。
        """
        shape_list = [tuple(shape) for shape in shape_list] if shape_list else []
        self._cache = ShapePerformanceCacheManager(
            ema_alpha=ema_alpha,
            shape_list=shape_list,
            shm_name=shm_name,
            create=False,
            lock=lock
        )

        self._logger = logger or logging.getLogger(__name__)

    def __del__(self):
        """
        析构函数
        """
        self._cache.release()

    def update(
        self,
        shape: Tuple[int, int, int],
        performance: float,
    ):
        """
        更新历史性能记录
        :param shape: 输入形状 (M, K, N)
        :param performance: 当前性能 (GFlops)
        """
        log_str = self._cache.update(shape=shape, performance=performance)
        if log_str:
            self._logger.info(log_str)

    def get_ema_performance(
        self,
        shape: Tuple[int, int, int]
    ) -> Optional[float]:
        """
        获取指定形状和线程配置的EMA性能（用作reward baseline）
        :param shape: 输入形状 (M, K, N)
        :return: EMA性能，未记录则返回None
        """
        performance = self._cache.get_shape_performances(shape=shape)
        return performance[2] if performance[2] > 0 else None

    def get_shape_best_performance(
        self,
        shape: Tuple[int, int, int],
    ) -> Optional[float]:
        """
        获取指定形状和线程配置的最好性能
        :param shape: 输入形状 (M, K, N)
        :return: 最好性能，未记录则返回None
        """
        performance = self._cache.get_shape_performances(shape=shape)
        return performance[1] if performance[1] > 0 else None

    def get_global_avg_performance(self) -> Optional[float]:
        """
        获取全局平均性能（所有shape和线程配置的所有采样的平均GFlops）
        用作首次遇到新shape和线程配置时的参考baseline
        :return: 全局平均GFlops，无记录则返回None
        """
        performance = self._cache.get_global_performances()
        return performance[3] if performance[3] > 0 else None

    def get_best_global_avg_performance(self) -> Optional[float]:
        """
        获取历史最好的全局平均性能
        这个值只增不减，避免了性能下降时基准过低的问题
        用作reward计算的基准，鼓励agent持续提升性能
        :return: 历史最好的全局平均GFlops，无记录则返回None
        """
        performance = self._cache.get_global_performances()
        return performance[2] if performance[2] > 0 else None
