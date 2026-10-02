# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:44
# Contact       : hucheng.lew@qq.com
# File          : __init__.py.py
# IDE           : PyCharm
# Description   :


from .layout import MatrixLayout, parse_layout
from .precision_type import ElementPrecision, PrecisionType, parse_precision
from .prefetch import PrefetchCacheLevel, parse_prefetch
from .scale_type import ResultScaleType


def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)  # Python 3.3+
        else:
            yield item


from .cpu_arch import CpuArch
