# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/9 16:26
# Contact       : hucheng.lew@qq.com
# File          : __init__.py.py
# IDE           : PyCharm
# Description   : 用于选择具体的算法实现的 ARM 矩阵乘法内核生成器


from .arm_neon_matul_kernel_with_outer_product_cr_ar_br_anp_bnp_f32_generator import ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator
from .arm_neon_matul_kernel_with_outer_product_cr_ar_br_anp_bfp_f32_generator import ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator
from gemm_codegen.kernel.gemm.base import (KernelParameters, KernelAlgorithm, MatrixLayout, BaseMatMulKernelGenerator, PrecisionType, PackingStrategy)


def kernel_generator_factory(params: KernelParameters) -> BaseMatMulKernelGenerator:
    """根据参数选择具体的 ARM 矩阵乘法内核生成器"""
    # 目前仅支持外积法实现
    if params.instruction_set == "neon":
        kernel_alg_type =  params.get_algorithm_type()
        if kernel_alg_type.value == KernelAlgorithm.OUTER_PRODUCT.value:
            if (params.c_layout.value == MatrixLayout.ROW_MAJOR.value) and (params.a_layout.value == MatrixLayout.ROW_MAJOR.value) and (params.b_layout.value == MatrixLayout.ROW_MAJOR.value):
                if params.data_type.precision.value == PrecisionType.FP32.value:
                    if (params.a_packing.value == PackingStrategy.NO_PACKING.value) and (params.b_packing.value == PackingStrategy.NO_PACKING.value):
                        return ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator(params)
                    elif (params.a_packing.value == PackingStrategy.NO_PACKING.value) and (params.b_packing.value == PackingStrategy.FINE_PACKING.value):
                        return ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator(params)
                    else:
                        raise NotImplementedError("当前仅支持 RowMajor C, RowMajor A, RowMajor B 矩阵乘法内核生成器，A 矩阵无打包，B 矩阵无打包或细粒度打包的情况")

    raise NotImplementedError("当前仅支持 ARM NEON 平台下，外积法实现的 RowMajor C, RowMajor A, RowMajor B 矩阵乘法内核生成器")
