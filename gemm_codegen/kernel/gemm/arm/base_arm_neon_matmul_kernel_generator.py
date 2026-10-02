# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/11 16:24
# Contact       : hucheng.lew@qq.com
# File          : base_neon_matmul_kernel_geerator.py
# IDE           : PyCharm
# Description   : ARM NEON 矩阵乘法内核生成器基类

from typing import List, Tuple, Set
from math import floor
from gemm_codegen.kernel.gemm.base import (
    BaseMatMulKernelGenerator, KernelParameters, AllocatedGeneralRegister, AllocatedVectorRegister
)

class BaseArmNeonMatMulKernelGenerator(BaseMatMulKernelGenerator):
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
            f"ldp x8, x9, [sp]        // 读取{self.register_allocation.stride_reg_ldb.var_name}, {self.register_allocation.prefetch_reg_c.var_name}",
            f"ldp x10, x11, [sp, #16]      // 读取{self.register_allocation.prefetch_reg_a.var_name}, {self.register_allocation.prefetch_reg_b.var_name}",
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


    def _check_inst_ready(self, inst: Tuple[
                                List[AllocatedVectorRegister | AllocatedGeneralRegister],
                                str,
                                List[AllocatedVectorRegister | AllocatedGeneralRegister],
                                List[AllocatedVectorRegister | AllocatedGeneralRegister],
                            ],
                          available_regs_set: Set[AllocatedVectorRegister | AllocatedGeneralRegister]) -> bool:
        """
        检查指令是否就绪
        :param inst: 指令, 指令是一个三元组 (调度前需就续的寄存器列表, 指令字符串, 调度后不可用的寄存器列表, 调度后可用的寄存器列表)
        :param available_regs_set: 就绪的寄存器
        :return: 是否就绪
        """
        for reg in inst[0]:
            if reg not in available_regs_set:
                return False

        for reg in inst[3]:
            if reg in available_regs_set:
                return False
        return True


    def _schedule_inst_seq(self,
                           inst_seqs: List[
                               List[
                                   Tuple[
                                       List[AllocatedVectorRegister | AllocatedGeneralRegister],
                                       str,
                                       List[AllocatedVectorRegister | AllocatedGeneralRegister],
                                       List[AllocatedVectorRegister | AllocatedGeneralRegister],
                                   ]
                               ]
                           ],
                           available_regs_set: Set[AllocatedVectorRegister | AllocatedGeneralRegister]) -> List[str]:
        """
        生成kernel的指令序列
        :param inst_seqs: 指令序列表, 每个元素是一个指令序列， 指令序列是指令的列表， 每条指令是一个三元组 (调度前需就续的寄存器列表, 指令字符串, 调度后不可用的寄存器列表, 调度后可用的寄存器列表)
        :param available_regs_set:      调度时可用的寄存器集合
        :return: 指令序列
        """

        kernel_inst_seq = []
        loop_index = 0
        while True:
            empty_inst_seq_num = 0
            for idx in range(len(inst_seqs)):
                inst_idx = (loop_index + idx) % len(inst_seqs)
                inst_seq = inst_seqs[inst_idx]
                if len(inst_seq) > 0:
                    inst = inst_seq[0]
                    check_flag = self._check_inst_ready(inst, available_regs_set)
                    if check_flag:
                        loop_index = (inst_idx + 1) % len(inst_seqs)
                        break
                else:
                    empty_inst_seq_num += 1
            else:
                if empty_inst_seq_num == len(inst_seqs):
                    break
                else:
                    raise Exception("No instruction is ready!!!")

            if check_flag:
                kernel_inst_seq.append(inst[1])
                # 更新可用寄存器集合
                for reg in inst[2]:
                    try:
                        available_regs_set.remove(reg)
                    except KeyError:
                        raise Exception(f"Register {reg.var_name} to be freed is not in available_regs_set")
                for reg in inst[3]:
                    available_regs_set.add(reg)
                inst_seq.pop(0)

        return kernel_inst_seq
