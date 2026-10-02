# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:48
# Contact       : hucheng.lew@qq.com
# File          : prefetch.py
# IDE           : PyCharm
# Description   : 定义预取缓存级别

from enum import Enum

class PrefetchCacheLevel(Enum):
    """预取缓存级别"""
    NONE = "None"
    L1D = "L1"
    L2 = "L2"
    L3 = "L3"


def parse_prefetch(prefetch_str: str) -> PrefetchCacheLevel:
    """解析预取策略"""
    if prefetch_str.lower() == 'none':
        return PrefetchCacheLevel.NONE
    elif prefetch_str.lower() == 'l1d':
        return PrefetchCacheLevel.L1D
    elif prefetch_str.lower() == 'l2':
        return PrefetchCacheLevel.L2
    elif prefetch_str.lower() == 'l3':
        return PrefetchCacheLevel.L3
    else:
        raise ValueError(f"不支持的预取策略: {prefetch_str}")
