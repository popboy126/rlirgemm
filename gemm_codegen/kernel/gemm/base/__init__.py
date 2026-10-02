# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/11 21:27
# Contact       : hucheng.lew@qq.com
# File          : __init__.py.py
# IDE           : PyCharm
# Description   : 用于代码生成的基础数据对象

from .kernel import (
    MatrixLayout,
    PackingStrategy,
    PrefetchCacheLevel,
    KernelAlgorithm,
    KernelShape,
    KernelShapes,
    KernelParameters
)

from .register_allocation import (
    AllocatedGeneralRegister,
    AllocatedGeneralRegisters,
    AllocatedVectorRegister,
    AllocatedVectorRegisters,
    RegisterAllocation
)

from .base_matmul_kernel_generator import BaseMatMulKernelGenerator

from gemm_codegen.common import ( PrecisionType, ElementPrecision, ResultScaleType)
