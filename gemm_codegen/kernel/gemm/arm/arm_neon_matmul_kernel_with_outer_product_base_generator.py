# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/11 16:27
# Contact       : hucheng.lew@qq.com
# File          : arm_neon_matmul_kernel_with_outer_product_base_generator.py
# IDE           : PyCharm
# Description   : 采用外积法实现的 ARM NEON 矩阵乘法内核生成器基类


from .base_arm_neon_matmul_kernel_generator import BaseArmNeonMatMulKernelGenerator
from gemm_codegen.kernel.gemm.base import KernelParameters


class ArmNeonMatMulKernelWithOuterProductBaseGenerator(BaseArmNeonMatMulKernelGenerator):
    """
    采用外积法实现的 ARM NEON 矩阵乘法内核生成器基类
    """
    def __init__(self, params: KernelParameters):
        super().__init__(params)
