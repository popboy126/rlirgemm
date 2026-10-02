# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/3/31 14:59
# Contact       : hucheng.lew@qq.com
# File          : FT2000GemmAgent.py
# IDE           : PyCharm
# Description   : FT2000的GEMM Agent


from .gemm_base_agent import GemmBaseAgent
import logging
from typing import Tuple, List

class FT2000GemmAgent(GemmBaseAgent):
    """
    FT2000+的GEMM Agent
    """
    def __init__(self, user, ip, port, password,
                 logfile_dir: str, logfile_name: str, logfile_level: int = logging.DEBUG):
        """
        初始化
        """
        super(FT2000GemmAgent, self).__init__(user, ip, port, password, logfile_dir, logfile_name, logfile_level)


    def get_l1d_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L1D缓存特征
        :return: (缓存行大小，缓存路数，缓存组数，缓存延迟(ns))
        """
        return 64, 2, 256, 1.819

    def get_l2_cache_features(self) -> Tuple[int, int, int, float]:
        """
        获取L2缓存特征
        :return: (缓存行大小，缓存组数，缓存路数，缓存延迟(ns))
        """
        return 64, 16, 2048, 4.175

    def get_fmla_features(self) -> Tuple[float, int]:
        """
        获取FMLA特征
        :return: (fmla指令的延迟(ns), fmla指令执行通量)
        """
        return 6 * (1 / 2.2), 2

    def get_mem_features(self) -> Tuple[float, float, float, float]:
        """
        获取内存特征
        :return: (内存按序读取延迟(ns)，内存随机读取延迟(ns)，内存读带宽(MB/s)，内存写带宽(MB/s))
        """
        return 10.4, 150.5, 6419.5, 8186
