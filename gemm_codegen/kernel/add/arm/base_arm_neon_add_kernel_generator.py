# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/29 09:08
# Contact       : hucheng.lew@qq.com
# File          : base_arm_neon_add_kernel_generator.py
# IDE           : PyCharm
# Description   : ARM NEON加法内核代码生成器基类


from typing import List
from math import floor
from gemm_codegen.kernel.add.base import (
    BaseAddKernelGenerator, KernelParameters
)

class BaseArmNeonAddKernelGenerator(BaseAddKernelGenerator):
    """ARM NEON平台矩阵乘法内核生成器"""

    def __init__(self, params: KernelParameters):
        super().__init__(params)

        # NEON寄存器配置
        self.vector_count = 32
        self.general_count = 29  # X0-X30 (X31是零寄存器或栈指针)

    def get_vector_register_count(self) -> int:
        return self.vector_count

    def get_general_register_count(self) -> int:
        return self.general_count

    def get_vector_register_names(self) -> List[str]:
        return [f"v{i}" for i in range(self.vector_count - 1, 0, -1)]  # 保留一个给alpha和beta

    def get_general_register_names(self) -> List[str]:
        return [f"x{i}" for i in range(self.general_count)]

    def _get_vector_width(self) -> int:
        return 128  # NEON是128-bit

    def _get_vector_byte_width(self) -> int:
        return floor(self._get_vector_width() / 8)

    def _get_instruction_set_macro(self) -> str:
        return "__ARM_NEON"

    def generate_function_prologue(self) -> List[str]:
        """生成函数序言

        ARM AArch64调用约定:
        - 参数: X0-X7, 然后栈
        - 返回值: X0
        - 保存: X19-X28, X29(FP), X30(LR)
        """
        instructions = [
            "// 读取x8-x13参数到寄存器",
            f"ldr {self.register_allocation.prefetch_reg_partial_sum.reg_name}, [sp]        // 读取{self.register_allocation.prefetch_reg_partial_sum.var_name}",
            "// 保存返回地址和帧指针",
            "stp x29, x30, [sp, #-16]!      // 保存当前返回地址和帧指针",
            "// 保存callee-saved寄存器",
            "stp x19, x20, [sp, #-16]!",
            "stp x21, x22, [sp, #-16]!",
            "stp x23, x24, [sp, #-16]!",
            "stp x25, x26, [sp, #-16]!",
            "stp x27, x28, [sp, #-16]!",
            "// 保存v8-v15寄存器",
            "stp q8, q9, [sp, #-32]!",
            "stp q10, q11, [sp, #-32]!",
            "stp q12, q13, [sp, #-32]!",
            "stp q14, q15, [sp, #-32]!",
        ]

        return instructions

    def generate_function_epilogue(self) -> List[str]:
        """生成函数尾声"""
        instructions = [
            "// 恢复v8-v15寄存器",
            "ldp q14, q15, [sp], #32",
            "ldp q12, q13, [sp], #32",
            "ldp q10, q11, [sp], #32",
            "ldp q8, q9, [sp], #32",
            "// 恢复callee-saved寄存器",
            "ldp x27, x28, [sp], #16",
            "ldp x25, x26, [sp], #16",
            "ldp x23, x24, [sp], #16",
            "ldp x21, x22, [sp], #16",
            "ldp x19, x20, [sp], #16",
            "ldp x29, x30, [sp], #16      // 恢复返回地址和帧指针",
        ]
        return instructions
