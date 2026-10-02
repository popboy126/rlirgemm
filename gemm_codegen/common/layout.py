# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:47
# Contact       : hucheng.lew@qq.com
# File          : layout.py
# IDE           : PyCharm
# Description   :  定义矩阵存储布局

from enum import Enum

class MatrixLayout(Enum):
    """矩阵存储布局"""
    ROW_MAJOR = "RowMajor"
    COL_MAJOR = "ColMajor"


def parse_layout(layout_str: str) -> MatrixLayout:
    """解析矩阵布局字符串"""
    if layout_str.lower() in ['rowmajor', 'row']:
        return MatrixLayout.ROW_MAJOR
    elif layout_str.lower() in ['colmajor', 'col']:
        return MatrixLayout.COL_MAJOR
    else:
        raise ValueError(f"不支持的矩阵布局: {layout_str}")
