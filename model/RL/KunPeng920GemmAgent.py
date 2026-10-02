# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/3/31 14:07
# Contact       : hucheng.lew@qq.com
# File          : KunPeng920GemmAgent.py
# IDE           : PyCharm
# Description   : KunPeng920的GEMM Agent

from gemm_base_agent import GemmBaseAgent
import logging
from typing import Tuple, Optional

class KunPeng920GemmAgent(GemmBaseAgent):
    """
    KunPeng920的GEMM Agent
    """
    def __init__(self, user, ip, port, password, agent_name: str, logger: Optional[logging.Logger]=None):
        """
        初始化
        """
        super(KunPeng920GemmAgent, self).__init__(user, ip, port, password, f"kp920_{agent_name}", logger)


    def get_l1d_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L1D缓存特征
        :return: (缓存行大小，缓存路数，缓存组数，缓存延迟(ns))
        """
        return 64, 4, 256, 1.539

    def get_l2_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L2缓存特征
        :return: (缓存行大小，缓存组数，缓存路数，缓存延迟(ns))
        """
        return 64, 8, 1024, 3.39425

    def get_fmla_features(self) -> Tuple[float, int]:
        """
        获取FMLA特征
        :return: (fmla指令的延迟(ns), fmla指令执行通量)
        """
        return 1.92, 2

    def get_mem_features(self) -> Tuple[float, float, float, float]:
        """
        获取内存特征
        :return: (内存按序读取延迟(ns)，内存随机读取延迟(ns)，内存读带宽(MB/s)，内存写带宽(MB/s))
        """
        return 8.385, 115.35, 9055.5, 4015.25



