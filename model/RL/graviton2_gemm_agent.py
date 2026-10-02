# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/10 22:28
# Contact       : hucheng.lew@qq.com
# File          : graviton2_gemm_agent.py
# IDE           : PyCharm
# Description   : graviton2平台的agent


from gemm_base_agent import GemmBaseAgent
import logging
from typing import Tuple, Optional

class Graviton2GemmAgent(GemmBaseAgent):
    """
    KunPeng920的GEMM Agent
    """
    def __init__(self, user, ip, port, password, agent_name: str, logger: Optional[logging.Logger]=None):
        """
        初始化
        """
        super(Graviton2GemmAgent, self).__init__(user, ip, port, password, f"graviton2_{agent_name}", logger)


    def get_l1d_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L1D缓存特征
        :return: (缓存行大小，缓存路数，缓存组数，缓存延迟(ns))
        """
        return 64, 4, 256, 1.602

    def get_l2_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L2缓存特征
        :return: (缓存行大小，缓存组数，缓存路数，缓存延迟(ns))
        """
        return 64, 8, 2048, 2.64775

    def get_fmla_features(self) -> Tuple[float, int]:
        """
        获取FMLA特征
        :return: (fmla指令的延迟(ns), fmla指令执行通量)
        """
        return 0.8, 2

    def get_mem_features(self) -> Tuple[float, float, float, float]:
        """
        获取内存特征
        :return: (内存按序读取延迟(ns)，内存随机读取延迟(ns)，内存读带宽(MB/s)，内存写带宽(MB/s))
        """
        return 7.0585, 145.975, 11000, 19550
