# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/17 14:25
# Contact       : hucheng.lew@qq.com
# File          : scale_type.py
# IDE           : PyCharm
# Description   : 缩放类型


from enum import Enum


class ResultScaleType(Enum):
    """结果缩放类型枚举"""
    ALPHA1_BETA0 = "alpha1_beta0"  # alpha=1, beta=0
    ALPHA1_BETA1 = "alpha1_beta1"  # alpha=1, beta=1
    ALPHA1_BETA = "alpha1_beta"  # alpha=1, beta=*
    ALPHA_BETA0 = "alpha_beta0"  # alpha=*, beta=0
    ALPHA_BETA1 = "alpha_beta1"  # alpha=*, beta=1
    ALPHA_BETA = "alpha_beta"    # alpha=*, beta=*
