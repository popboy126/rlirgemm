# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/9 16:39
# Contact       : hucheng.lew@qq.com
# File          : __init__.py.py
# IDE           : PyCharm
# Description   :
"""Test package bootstrap for the generator's historical absolute imports."""

from pathlib import Path
import sys

_CODEGEN_ROOT = Path(__file__).resolve().parents[1]
if str(_CODEGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEGEN_ROOT))
