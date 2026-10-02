# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:31
# Contact       : hucheng.lew@qq.com
# File          : __init__.py.py
# IDE           : PyCharm
# Description   :


from .arm_neon_add_kernel_cr_f32_generator import ArmNeonAddKernelCrF32Generator
from gemm_codegen.kernel.add.base import (KernelParameters, BaseAddKernelGenerator)
from gemm_codegen.common import MatrixLayout, PrecisionType


def kernel_generator_factory(params: KernelParameters) -> BaseAddKernelGenerator:
    """根据参数选择具体的 ARM 矩阵乘法内核生成器"""
    if params.instruction_set == "neon":
        if (params.layout.value == MatrixLayout.ROW_MAJOR.value):
            if params.data_type.precision.value == PrecisionType.FP32.value:
                return ArmNeonAddKernelCrF32Generator(params)

    raise NotImplementedError("当前仅支持 ARM NEON 平台下，实现的 RowMajor C, RowMajor Partial Sum 矩阵加法内核生成器")
