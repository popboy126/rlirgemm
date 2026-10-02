# -*- coding: utf-8 -*-
import json
from copy import deepcopy
from itertools import chain, pairwise, zip_longest
from math import ceil, floor
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from gemm_codegen.kernel.gemm.base import (
    AllocatedGeneralRegister,
    AllocatedGeneralRegisters,
    AllocatedVectorRegister,
    AllocatedVectorRegisters,
    KernelParameters,
    MatrixLayout,
    PackingStrategy,
    PrecisionType,
    PrefetchCacheLevel,
    RegisterAllocation,
    ResultScaleType,
)

# Author        : huche
# Create Time   : 2026/1/2 14:50
# Contact       : hucheng.lew@qq.com
# File          : arm_neon_matul_kernel_with_outer_product_cr_ar_br_anp_bfp_f32_generator.py
# IDE           : PyCharm
# Description   :
from .arm_neon_matul_kernel_with_outer_product_cr_ar_br_anp_bnp_f32_generator import (
    ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator,
)


class ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator(
    ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator
):
    """ARM NEON 平台下，外积法实现的 RowMajor C, RowMajor A, RowMajor B 矩阵乘法内核生成器，A 矩阵无打包，B 矩阵细粒度打包，数据类型为 FP32"""

    def __init__(self, params: KernelParameters):
        super().__init__(params)

        if self.params.data_type.precision.value != PrecisionType.FP32.value:
            raise ValueError(
                "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator 仅支持 f32 数据类型"
            )

        if not (
            (self.params.c_layout.value == MatrixLayout.ROW_MAJOR.value)
            and (self.params.a_layout.value == MatrixLayout.ROW_MAJOR.value)
            and (self.params.b_layout.value == MatrixLayout.ROW_MAJOR.value)
        ):
            raise ValueError(
                "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator 仅支持 C、A、B 均为行主序布局"
            )

        if (self.params.a_packing.value != PackingStrategy.NO_PACKING.value) or (
            self.params.b_packing.value != PackingStrategy.FINE_PACKING.value
        ):
            raise ValueError(
                "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBfpF32Generator 仅支持A无打包, B细粒度打包"
            )

    def allocate_registers(self) -> RegisterAllocation:
        """
        为外层乘积分配寄存器
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        # 分配C的向量寄存器
        c_vec_regs_needed = self.params.mr * ceil(
            self.params.nr / elements_per_vector_reg
        )
        # 分配A的向量寄存器

        b_row_regs_needed = ceil(self.params.nr / elements_per_vector_reg)
        min_b_row_regs_needed = (
            b_row_regs_needed
            if b_row_regs_needed > self.params.vec_reg_reuse_interval
            else self.params.vec_reg_reuse_interval
        )
        a_vec_regs_needed = (
            self.params.mr
            if (self.params.mr - 1) * min_b_row_regs_needed
            >= self.params.vec_reg_reuse_interval
            else self.params.vec_reg_reuse_interval
        )  # 支持矩阵A寄存器重用的最小数量
        b_vec_regs_needed = (self.get_vector_register_count() - 1) - (
            c_vec_regs_needed + a_vec_regs_needed
        )
        # 向量寄存器分配生成器
        vector_register_allocator = self.vector_register_allocator()
        general_register_allocator = self.general_register_allocator()
        return RegisterAllocation(
            c_vector_regs=AllocatedVectorRegisters(
                mat_name="C",
                reg_names=list(
                    reversed(
                        [
                            next(vector_register_allocator)
                            for _ in range(c_vec_regs_needed)
                        ]
                    )
                ),
                precision=self.params.data_type,
            ),
            a_vector_regs=AllocatedVectorRegisters(
                mat_name="A",
                reg_names=list(
                    reversed(
                        [
                            next(vector_register_allocator)
                            for _ in range(a_vec_regs_needed)
                        ]
                    )
                ),
                precision=self.params.data_type,
            ),
            b_vector_regs=AllocatedVectorRegisters(
                mat_name="B",
                reg_names=list(
                    reversed(
                        [
                            next(vector_register_allocator)
                            for _ in range(b_vec_regs_needed)
                        ]
                    )
                ),
                precision=self.params.data_type,
            ),
            # 注意通用寄存器的分配要跟函数的调用规则匹配上
            pointer_reg_c=AllocatedGeneralRegister(
                var_name="C",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_a=AllocatedGeneralRegister(
                var_name="A",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_b=AllocatedGeneralRegister(
                var_name="B",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            dim_reg_m=AllocatedGeneralRegister(
                var_name="M",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            dim_reg_n=AllocatedGeneralRegister(
                var_name="N",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            dim_reg_k=AllocatedGeneralRegister(
                var_name="K",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            stride_reg_ldc=AllocatedGeneralRegister(
                var_name="ldc",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            stride_reg_lda=AllocatedGeneralRegister(
                var_name="lda",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            stride_reg_ldb=AllocatedGeneralRegister(
                var_name="ldb",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            prefetch_reg_c=AllocatedGeneralRegister(
                var_name="prefetch_c_stride",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            prefetch_reg_a=AllocatedGeneralRegister(
                var_name="prefetch_a_stride",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            prefetch_reg_b=AllocatedGeneralRegister(
                var_name="prefetch_b_stride",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_bc=AllocatedGeneralRegister(
                var_name="bc",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            scalar_reg_alpha=AllocatedVectorRegister(
                var_name="alpha",
                reg_name="v0",
                precision=self.params.data_type,
                position=2,
            ),
            scalar_reg_beta=AllocatedVectorRegister(
                var_name="beta",
                reg_name="v0",
                precision=self.params.data_type,
                position=3,
            ),
            pointer_reg_c_for_loop_n=AllocatedGeneralRegister(
                var_name="c_block_addr_for_loop_n",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_c_for_current_block_base=AllocatedGeneralRegister(
                var_name="c_block_base",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_a_for_current_block_base=AllocatedGeneralRegister(
                var_name="a_block_base",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_b_for_current_block_base=AllocatedGeneralRegister(
                var_name="b_block_base",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            loop_reg_dealt_m=AllocatedGeneralRegister(
                var_name="dealt_m",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            loop_reg_dealt_n=AllocatedGeneralRegister(
                var_name="dealt_n",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            loop_reg_dealt_k=AllocatedGeneralRegister(
                var_name="dealt_k",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            pointer_reg_bc_for_loop_k=AllocatedGeneralRegister(
                var_name="bc_addr_for_loop_k",
                reg_name=next(general_register_allocator),
                spilled=False,
                offset=0,
            ),
            temp_regs=AllocatedGeneralRegisters(
                regs=[
                    AllocatedGeneralRegister(
                        var_name=f"temp_reg_{idx}",
                        reg_name=reg,
                        spilled=False,
                        offset=0,
                    )
                    for idx, reg in enumerate(general_register_allocator)
                ]  # 剩余的全部分配
            ),
        )

    def generate_function_prologue(self) -> List[str]:
        """生成函数序言

        ARM AArch64调用约定:
        - 参数: X0-X7, 然后栈
        - 返回值: X0
        - 保存: X19-X28, X29(FP), X30(LR)
        """
        instructions = [
            "// 读取x8-x13参数到寄存器",
            f"ldp {self.register_allocation.stride_reg_ldb.reg_name}, {self.register_allocation.prefetch_reg_c.reg_name}, [sp]        // 读取{self.register_allocation.stride_reg_ldb.var_name}, {self.register_allocation.prefetch_reg_c.var_name}",
            f"ldp {self.register_allocation.prefetch_reg_a.reg_name}, {self.register_allocation.prefetch_reg_b.reg_name}, [sp, #16]      // 读取{self.register_allocation.prefetch_reg_a.var_name}, {self.register_allocation.prefetch_reg_b.var_name}",
            f"ldr {self.register_allocation.pointer_reg_bc.reg_name}, [sp, #32]      // 读取bc指针",
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
            f"mov {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}], v0.s[0]         // 将alpha的值复制到{self.register_allocation.scalar_reg_alpha.v_reg_name}",
            f"mov {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}], v1.s[0]           // 将beta的值复制到{self.register_allocation.scalar_reg_beta.v_reg_name}",
        ]

        return instructions

    def generate_loop_m_with_packing(
        self,
        mr_temp_reg: AllocatedGeneralRegister,
        loop_m_macro_fmt: Template,
        loop_k_macro_fmt: Template,
        loop_k_packing_macro_fmt: Template,
    ) -> List[Dict[str, List[str] | Any]]:
        """生成M维度循环"""
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        nr_list = self.params.get_nr_list()
        loop_m_macros = list()
        for nr in nr_list:
            loop_m_instructions = []
            loop_m_instructions.append(
                current_indent
                + f"// M维度循环体: 处理mr={self.params.mr}, nr={nr}, kr={self.params.kr}"
            )
            loop_m_instructions.append(
                current_indent + f"// 实现mr={self.params.mr}, nr={nr} 的M循环计算逻辑"
            )
            loop_m_instructions.append(
                current_indent
                + f"eor {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}  // 清零M维度计数器"
            )
            mr_list = self.params.get_mr_list(nr)
            mr_list.append(0)  # 方便生成最后一个循环
            # 先生成打包所需的逻辑指令
            loop_m_instructions.append(
                f"LOOP_NR{nr}_LOOP_MR_PACKING_{label_suffix}:    // M维度打包计算过程"
            )
            for mr, next_mr in pairwise(mr_list):
                loop_m_instructions.append(
                    current_indent
                    + f"sub {mr_temp_reg.reg_name}, {self.register_allocation.dim_reg_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}   // 计算剩余未处理的M维度"
                )
                loop_m_instructions.append(
                    f"LOOP_NR{nr}_LOOP_MR{mr}_PACKING_START_{label_suffix}:         // 处理mr={mr}的M循环开始, 此时nr={mr}"
                )
                loop_m_instructions.append(
                    current_indent
                    + f"cmp {mr_temp_reg.reg_name}, #{mr}       // 比较未处理M维度与当前mr={mr}"
                )
                if next_mr != 0:
                    loop_m_instructions.append(
                        current_indent
                        + f"blt LOOP_NR{nr}_LOOP_MR{next_mr}_PACKING_START_{label_suffix}   // 如果未处理M维度小于当前mr={mr}，跳转到下一个mr={next_mr}循环, 此时nr={mr}"
                    )
                # 计算矩阵C的块的基地址
                loop_m_instructions.append(
                    current_indent
                    + f"madd {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.pointer_reg_c_for_loop_n.reg_name}   // 计算C块基地址"
                )
                # 计算矩阵A的块的基地址
                loop_m_instructions.append(
                    current_indent
                    + f"madd {self.register_allocation.pointer_reg_a_for_current_block_base.reg_name}, {self.register_allocation.stride_reg_lda.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.pointer_reg_a.reg_name}   // 计算A块M维度的基地址"
                )
                # 调用M循环体
                loop_m_instructions.append(
                    current_indent
                    + f"{loop_k_packing_macro_fmt.substitute(nr=nr, mr=mr)}            // 调用K维度打包循环体，处理nr={nr}, mr={mr}"
                )
                # 记录处理过程
                # 更新已处理的M维度计数器
                loop_m_instructions.append(
                    current_indent
                    + f"add {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, #{mr}   // 更新已处理的M维度计数器"
                )
                if next_mr != 0:
                    loop_m_instructions.append(
                        current_indent
                        + f"b LOOP_NR{nr}_LOOP_MR_MAIN_{label_suffix}   // 跳到主力计算核入口, 此时nr={nr}"
                    )

            loop_m_instructions.append(current_indent + "// 开始进入M维度主计算过程")
            loop_m_instructions.append(
                f"LOOP_NR{nr}_LOOP_MR_MAIN_{label_suffix}:    // M维度主计算过程"
            )
            for mr, next_mr in pairwise(mr_list):
                # 计算当前剩余未处理的M维度大小
                loop_m_instructions.append(
                    f"LOOP_NR{nr}_LOOP_MR{mr}_UPDATE_{label_suffix}:    // 更新剩余M维度计数器"
                )
                loop_m_instructions.append(
                    current_indent
                    + f"sub {mr_temp_reg.reg_name}, {self.register_allocation.dim_reg_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}   // 计算剩余未处理的M维度"
                )
                loop_m_instructions.append(
                    f"LOOP_NR{nr}_LOOP_MR{mr}_START_{label_suffix}:         // 处理mr={mr}的M循环开始, 此时nr={mr}"
                )
                loop_m_instructions.append(
                    current_indent
                    + f"cmp {mr_temp_reg.reg_name}, #{mr}       // 比较未处理M维度与当前mr={mr}"
                )
                if next_mr != 0:
                    loop_m_instructions.append(
                        current_indent
                        + f"blt LOOP_NR{nr}_LOOP_MR{next_mr}_START_{label_suffix}   // 如果未处理M维度小于当前mr={mr}，跳转到下一个mr={next_mr}循环, 此时nr={mr}"
                    )
                else:
                    loop_m_instructions.append(
                        current_indent
                        + f"blt LOOP_NR{nr}_LOOP_MR_END_{label_suffix}   // 如果未处理M维度小于mr={mr}，nr下又没有更小的mr，那么跳转到M循环结束, 此时nr={mr}"
                    )
                # 计算矩阵C的块的基地址
                loop_m_instructions.append(
                    current_indent
                    + f"madd {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.pointer_reg_c_for_loop_n.reg_name}   // 计算C块基地址"
                )
                # 计算矩阵A的块的基地址
                loop_m_instructions.append(
                    current_indent
                    + f"madd {self.register_allocation.pointer_reg_a_for_current_block_base.reg_name}, {self.register_allocation.stride_reg_lda.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.pointer_reg_a.reg_name}   // 计算A块M维度的基地址"
                )
                # 调用M循环体
                loop_m_instructions.append(
                    current_indent
                    + f"{loop_k_macro_fmt.substitute(nr=nr, mr=mr)}            // 调用K维度循环体，处理nr={nr}, mr={mr}"
                )
                # 记录处理过程
                # 更新已处理的M维度计数器
                loop_m_instructions.append(
                    current_indent
                    + f"add {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, #{mr}   // 更新已处理的M维度计数器"
                )
                loop_m_instructions.append(
                    current_indent
                    + f"b LOOP_NR{nr}_LOOP_MR{mr}_UPDATE_{label_suffix}    // 回到当前mr={mr}循环开始处, 此时nr={nr}"
                )

            loop_m_instructions.append(
                f"LOOP_NR{nr}_LOOP_MR_END_{label_suffix} :           // M循环结束, 此时nr={nr}"
            )
            loop_m_macros.append(
                {
                    "name": loop_m_macro_fmt.substitute(nr=nr),
                    "instructions": loop_m_instructions,
                }
            )
        return loop_m_macros

    def generate_loop_k_with_packing(
        self,
        k_temp_reg: AllocatedGeneralRegister,
        loop_k_packing_macro_fmt: Template,
        packing_kernel_macro_fmt: Template,
        load_c_macro_fmt: Template,
        store_c_macro_fmt: Template,
    ) -> List[Dict[str, List[str] | Any]]:
        """生成K维度打包循环"""
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        loop_k_macros = list()
        nr_list = self.params.get_nr_list()
        for nr in nr_list:
            mr_list = self.params.get_mr_list(nr)
            for mr in mr_list:
                loop_k_instructions = []
                loop_k_instructions.append(
                    current_indent
                    + f"// K维度打包循环体: 处理mr={mr}, nr={nr}, kr={self.params.kr}"
                )
                loop_k_instructions.append(
                    current_indent + f"// 实现mr={mr}, nr={nr} 的K维度打包循环计算逻辑"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"{load_c_macro_fmt.substitute(nr=nr, mr=mr)}        // 加载C矩阵块"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"eor {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}  // 清零K维度计数器"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"mov {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}   // 初始化剩余K维度计数器"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"mov {self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}, {self.register_allocation.pointer_reg_bc.reg_name}   // 初始化BC打包块地址寄存器"
                )
                kr_list = self.params.get_kr_list(mr=mr, nr=nr)
                kr_list.append(0)  # 方便生成最后一个循环
                for kr, next_kr in pairwise(kr_list):
                    loop_k_instructions.append(
                        f"LOOP_NR{nr}_MR{mr}_LOOP_KR{kr}_PACKING_START_{label_suffix} :         // 处理kr={kr}的K循环打包开始, 此时nr={nr}, mr={mr}"
                    )
                    loop_k_instructions.append(
                        current_indent
                        + f"cmp {k_temp_reg.reg_name}, #{kr}       // 比较未处理K维度与当前kr={kr}"
                    )
                    if next_kr != 0:
                        loop_k_instructions.append(
                            current_indent
                            + f"blt LOOP_NR{nr}_MR{mr}_LOOP_KR{next_kr}_PACKING_START_{label_suffix}    // 如果未处理K维度小于当前kr={kr}，跳转到下一个kr={next_kr}循环, 此时nr={nr}, mr={mr}"
                        )
                    else:
                        loop_k_instructions.append(
                            current_indent
                            + f"blt LOOP_NR{nr}_MR{mr}_LOOP_KR_PACKING_END_{label_suffix}    // 如果未处理K维度小于kr={kr}，nr下又没有更小的kr，那么跳转到K循环结束, 此时nr={nr}, mr={mr}"
                        )
                    # 调用K循环体
                    loop_k_instructions.append(
                        current_indent
                        + f"{packing_kernel_macro_fmt.substitute(nr=nr, mr=mr, kr=kr)}            // 调用打包内核计算体，处理nr={nr}, mr={mr}, kr={kr}"
                    )
                    # 记录处理过程
                    # 计算当前剩余未处理的K维度大小
                    loop_k_instructions.append(
                        current_indent
                        + f"sub {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 计算剩余未处理的K维度"
                    )
                loop_k_instructions.append(
                    f"LOOP_NR{nr}_MR{mr}_LOOP_KR_PACKING_END_{label_suffix} :           // K循环打包结束, 此时nr={nr}, mr={mr}"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"{store_c_macro_fmt.substitute(nr=nr, mr=mr)}        // 保存C矩阵块"
                )
                loop_k_macros.append(
                    {
                        "name": loop_k_packing_macro_fmt.substitute(nr=nr, mr=mr),
                        "instructions": loop_k_instructions,
                    }
                )
        return loop_k_macros

    def generate_loop_k_use_packing(
        self,
        k_temp_reg: AllocatedGeneralRegister,
        loop_k_macro_fmt: Template,
        kernel_macro_fmt: Template,
        load_c_macro_fmt: Template,
        store_c_macro_fmt: Template,
    ) -> List[Dict[str, List[str] | Any]]:
        """生成K维度循环"""
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        loop_k_macros = list()
        nr_list = self.params.get_nr_list()
        for nr in nr_list:
            mr_list = self.params.get_mr_list(nr)
            for mr in mr_list:
                loop_k_instructions = []
                loop_k_instructions.append(
                    current_indent
                    + f"// K维度循环体: 处理mr={mr}, nr={nr}, kr={self.params.kr}"
                )
                loop_k_instructions.append(
                    current_indent + f"// 实现mr={mr}, nr={nr} 的K循环计算逻辑"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"{load_c_macro_fmt.substitute(nr=nr, mr=mr)}        // 加载C矩阵块"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"eor {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}  // 清零K维度计数器"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"mov {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}   // 初始化剩余K维度计数器"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"mov {self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}, {self.register_allocation.pointer_reg_bc.reg_name}   // 初始化BC打包块地址寄存器"
                )
                kr_list = self.params.get_kr_list(mr=mr, nr=nr)
                kr_list.append(0)  # 方便生成最后一个循环
                for kr, next_kr in pairwise(kr_list):
                    loop_k_instructions.append(
                        f"LOOP_NR{nr}_MR{mr}_LOOP_KR{kr}_START_{label_suffix} :         // 处理kr={kr}的K循环开始, 此时nr={nr}, mr={mr}"
                    )
                    loop_k_instructions.append(
                        current_indent
                        + f"cmp {k_temp_reg.reg_name}, #{kr}       // 比较未处理K维度与当前kr={kr}"
                    )
                    if next_kr != 0:
                        loop_k_instructions.append(
                            current_indent
                            + f"blt LOOP_NR{nr}_MR{mr}_LOOP_KR{next_kr}_START_{label_suffix}    // 如果未处理K维度小于当前kr={kr}，跳转到下一个kr={next_kr}循环, 此时nr={nr}, mr={mr}"
                        )
                    else:
                        loop_k_instructions.append(
                            current_indent
                            + f"blt LOOP_NR{nr}_MR{mr}_LOOP_KR_END_{label_suffix}    // 如果未处理K维度小于kr={kr}，nr下又没有更小的kr，那么跳转到K循环结束, 此时nr={nr}, mr={mr}"
                        )
                    # 调用K循环体
                    loop_k_instructions.append(
                        current_indent
                        + f"{kernel_macro_fmt.substitute(nr=nr, mr=mr, kr=kr)}            // 调用内核计算体，处理nr={nr}, mr={mr}, kr={kr}"
                    )
                    # 记录处理过程
                    # 计算当前剩余未处理的K维度大小
                    loop_k_instructions.append(
                        current_indent
                        + f"sub {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 计算剩余未处理的K维度"
                    )
                loop_k_instructions.append(
                    f"LOOP_NR{nr}_MR{mr}_LOOP_KR_END_{label_suffix} :           // K循环结束, 此时nr={nr}, mr={mr}"
                )
                loop_k_instructions.append(
                    current_indent
                    + f"{store_c_macro_fmt.substitute(nr=nr, mr=mr)}        // 保存C矩阵块"
                )
                loop_k_macros.append(
                    {
                        "name": loop_k_macro_fmt.substitute(nr=nr, mr=mr),
                        "instructions": loop_k_instructions,
                    }
                )
        return loop_k_macros

    def _packing_mat_b_data_in_kernel_compute(
        self,
        kr: int,
        nr: int,
        mat_b_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        round_idx: int,
    ) -> List[
        Tuple[
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            str,
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
        ]
    ]:
        """
        生成内核计算体中装载矩阵B数据的指令
        @return: 指令列表, 每条指令为一个元组，包含(调度前需就续的寄存器列表, 指令字符串, 调度后不可用的寄存器列表, 调度后可用的寄存器列表)
        """
        if round_idx >= len(mat_b_vec_regs_scheduler):
            raise ValueError(f"round_idx={round_idx} 超出矩阵B向量寄存器调度器范围")
        instructions = []
        current_indent = "    "
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        round_scheduler = mat_b_vec_regs_scheduler[round_idx]
        instructions.append(
            (
                [],
                current_indent + f"// 打包矩阵B第{round_idx}轮的数据: kr={kr}, nr={nr}",
                [],
                [],
            )
        )
        remained_nr = nr
        while remained_nr > 0:
            stored_nr = nr - remained_nr
            c = ceil(stored_nr / elements_per_vector_reg)
            if remained_nr >= (2 * elements_per_vector_reg):
                instructions.append(
                    (
                        [
                            round_scheduler[c],
                            round_scheduler[c + 1],
                            self.register_allocation.pointer_reg_bc_for_loop_k,
                        ],
                        current_indent
                        + f"stp {round_scheduler[c].q_reg_name}, {round_scheduler[c + 1].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{2 * self._get_vector_byte_width()}   // 打包矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        # [round_scheduler[c], round_scheduler[c+1]],
                        [],  # 留给fmla指令清除
                        [],
                    )
                )
                remained_nr -= 2 * elements_per_vector_reg
            elif remained_nr >= elements_per_vector_reg:
                instructions.append(
                    (
                        [
                            round_scheduler[c],
                            self.register_allocation.pointer_reg_bc_for_loop_k,
                        ],
                        current_indent
                        + f"str {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 打包矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        # [round_scheduler[c]],
                        [],  # 留给fmla指令清除
                        [],
                    )
                )
                remained_nr -= elements_per_vector_reg
            elif remained_nr == 3:
                inst = (
                    current_indent
                    + f"str {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 打包矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 2}列对应的向量寄存器"
                )
                instructions.append(
                    (
                        [
                            round_scheduler[c],
                            self.register_allocation.pointer_reg_bc_for_loop_k,
                        ],
                        inst,
                        # [round_scheduler[c]],
                        [],  # 留给fmla指令清除
                        [],
                    )
                )
                remained_nr -= 3
            elif remained_nr == 2:
                inst = (
                    current_indent
                    + f"str {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 打包矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 - 1}对应的列向量寄存器, 多的会在存储时剔除"
                )
                instructions.append(
                    (
                        [
                            round_scheduler[c],
                            self.register_allocation.pointer_reg_bc_for_loop_k,
                        ],
                        inst,
                        # [round_scheduler[c]],
                        [],  # 留给fmla指令清除
                        [],
                    )
                )
                remained_nr -= 2
            elif remained_nr == 1:
                inst = (
                    current_indent
                    + f"str {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 打包矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 - 1}对应的列向量寄存器, 多的会在存储时剔除"
                )
                instructions.append(
                    (
                        [
                            round_scheduler[c],
                            self.register_allocation.pointer_reg_bc_for_loop_k,
                        ],
                        inst,
                        # [round_scheduler[c]],
                        [],  # 留给fmla指令清除
                        [],
                    )
                )
                remained_nr -= 1
            else:
                raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的打包")
        return instructions

    def _compute_result_in_kernel_compute(
        self,
        mr: int,
        kr: int,
        nr: int,
        mat_a_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        mat_b_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        round_idx: int,
    ) -> List[
        Tuple[
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            str,
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
        ]
    ]:
        """
        生成内核计算体中计算结果的指令
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        mat_a_round_idx, mat_a_round_lane_idx = divmod(
            round_idx, elements_per_vector_reg
        )
        if mat_a_round_idx >= len(mat_a_vec_regs_scheduler):
            raise ValueError(
                f"round_idx={mat_a_round_idx} 超出矩阵A向量寄存器调度器范围"
            )
        if round_idx >= len(mat_b_vec_regs_scheduler):
            raise ValueError(f"round_idx={round_idx} 超出矩阵B向量寄存器调度器范围")
        instructions = []
        current_indent = "    "
        instructions.append(
            (
                [],
                current_indent
                + f"// 计算开始: mr={mr}, kr={kr}, nr={nr}, round_idx={round_idx}",
                [],
                [],
            )
        )
        mat_a_round_scheduler = mat_a_vec_regs_scheduler[mat_a_round_idx]
        mat_b_round_scheduler = mat_b_vec_regs_scheduler[round_idx]
        result_rows, result_cols = mr, ceil(nr / elements_per_vector_reg)
        # 计算C矩阵的对应向量寄存器
        c_vec_regs, _ = self.register_allocation.c_vector_regs.get_rearranged_reg_names(
            result_rows, result_cols
        )
        # 计算逻辑
        for r in range(mr):
            a_vec_reg = mat_a_round_scheduler[r]
            mat_a_data = f"{a_vec_reg.v_reg_name}.s[{mat_a_round_lane_idx}]"
            for c in range(len(mat_b_round_scheduler)):
                c_vec_reg = c_vec_regs[r][c]
                b_vec_reg = mat_b_round_scheduler[c]
                instructions.append(
                    (
                        [c_vec_reg, a_vec_reg, b_vec_reg],
                        current_indent
                        + f"fmla {c_vec_reg.v_reg_name}.4s, {b_vec_reg.v_reg_name}.4s, {mat_a_data}   // 计算C[{r},{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 1}] += A[{r},{round_idx}] * B[{round_idx},{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 1}]",
                        [],
                        [],
                    )
                )

                if r == (mr - 1):
                    # 可以在计算后立即释放矩阵B的向量寄存器
                    instructions.append(
                        (
                            [],
                            current_indent
                            + f"// 矩阵B第{round_idx}轮的向量寄存器{b_vec_reg.v_reg_name}已经使用完毕，可以释放掉",
                            [b_vec_reg],
                            [],
                        )
                    )

            if (mat_a_round_lane_idx == (elements_per_vector_reg - 1)) or (
                round_idx == kr - 1
            ):  # 当K维度不够四个元素时，后续的计算不需要了
                # 当前轮次的最后一个lane，说明矩阵A的当前向量寄存器已经用完，可以释放掉
                instructions.append(
                    (
                        [a_vec_reg],
                        current_indent
                        + f"// 矩阵A第{mat_a_round_idx}轮的向量寄存器{a_vec_reg.v_reg_name}已经使用完毕，可以释放掉",
                        [a_vec_reg],
                        [],
                    )
                )
        # else:
        #     instructions.append(
        #         (
        #             [],
        #             current_indent + f"// 计算开结束: mr={mr}, kr={kr}, nr={nr}, round_idx={round_idx}",
        #             deepcopy(mat_b_round_scheduler),
        #             []
        #         )
        #     )
        return instructions

    def _load_mat_a_data_in_kernel_compute_with_alpha_scale(
        self,
        mr: int,
        kr: int,
        a_row_addr_temp_regs: List[AllocatedGeneralRegister],
        mat_a_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        round_idx: int,
    ) -> List[
        Tuple[
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            str,
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
        ]
    ]:
        """
        生成内核计算体中装载矩阵A数据的指令
        """
        if round_idx >= len(mat_a_vec_regs_scheduler):
            raise ValueError(f"round_idx={round_idx} 超出矩阵A向量寄存器调度器范围")
        instructions = []
        current_indent = "    "
        alpha_beta_type = self.get_result_scale_type()
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        instructions.append(
            (
                [],
                current_indent + f"// 装载矩阵A第{round_idx}轮的数据: mr={mr}, kr={kr}",
                [],
                [],
            )
        )
        round_scheduler = mat_a_vec_regs_scheduler[round_idx]
        remained_kr = kr - round_idx * elements_per_vector_reg
        for r in range(len(round_scheduler)):
            a_row_addr_temp_reg = a_row_addr_temp_regs[r]
            if remained_kr >= 4:
                inst_str = (
                    current_indent
                    + f"ldr {round_scheduler[r].q_reg_name}, [{a_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载矩阵A第{r}行, 第 0 ~ {elements_per_vector_reg - 1}列对应的向量寄存器"
                )
            elif remained_kr == 3:
                inst_str = (
                    current_indent
                    + f"ldr {round_scheduler[r].q_reg_name}, [{a_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载矩阵A第{r}行, 第 0 ~ {elements_per_vector_reg - 2}列对应的向量寄存器"
                )
                inst_str += (
                    "\n"
                    + current_indent
                    + f"ins {round_scheduler[r].v_reg_name}.s[3], {round_scheduler[r].v_reg_name}.s[0]    // 用矩阵A第{r}行, 第0列的元素填充一下，避免计算出错"
                )
            elif remained_kr == 2:
                inst_str = (
                    current_indent
                    + f"movi {round_scheduler[r].v_reg_name}.4s, #0   // 先清零向量寄存器"
                )
                inst_str += (
                    "\n"
                    + current_indent
                    + f"ldr {round_scheduler[r].d_reg_name}, [{a_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 装载矩阵A第{r}行, 第 0 ~ 1 对应的列向量寄存器"
                )
            elif remained_kr == 1:
                inst_str = (
                    current_indent
                    + f"movi {round_scheduler[r].v_reg_name}.4s, #0   // 先清零向量寄存器"
                )
                inst_str += (
                    "\n"
                    + current_indent
                    + f"ldr {round_scheduler[r].s_reg_name}, [{a_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 装载矩阵A第{r}行, 第0 ~ 0对应的列向量寄存器"
                )
            else:
                raise NotImplementedError(f"不支持的剩余kr={remained_kr}元素数的加载")

            if alpha_beta_type.value in [
                ResultScaleType.ALPHA_BETA0.value,
                ResultScaleType.ALPHA_BETA1.value,
                ResultScaleType.ALPHA_BETA.value,
            ]:
                inst_str += (
                    "\n"
                    + current_indent
                    + f"\n{current_indent}fmul {round_scheduler[r].v_reg_name}.4s, {round_scheduler[r].v_reg_name}.4s, {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]             // 矩阵A第{r}行, 第 0 ~ {elements_per_vector_reg - 1}列对应的向量寄存器乘以alpha"
                )
            instructions.append(
                ([a_row_addr_temp_reg], inst_str, [], [round_scheduler[r]])
            )

            if (((round_idx + 1) % 4) == 0) and (
                self.params.a_prefetch != PrefetchCacheLevel.NONE.value
            ):
                instructions.append(
                    (
                        [a_row_addr_temp_reg],
                        current_indent
                        + f"prfm PLD{self.params.a_prefetch.value.upper()}STRM, [{a_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_a.reg_name}]   // 预取A矩阵数据",
                        [],
                        [],
                    )
                )
        return instructions

    def _load_mat_b_data_in_kernel_compute(
        self,
        kr: int,
        nr: int,
        b_row_addr_temp_regs: List[AllocatedGeneralRegister],
        mat_b_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        round_idx: int,
    ) -> List[
        Tuple[
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            str,
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
        ]
    ]:
        """
        生成内核计算体中装载矩阵B数据的指令
        """
        if round_idx >= len(mat_b_vec_regs_scheduler):
            raise ValueError(f"round_idx={round_idx} 超出矩阵B向量寄存器调度器范围")
        instructions = []
        current_indent = "    "
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        round_scheduler = mat_b_vec_regs_scheduler[round_idx]
        mat_b_rows, mat_b_cols = kr, len(round_scheduler)
        instructions.append(
            (
                [],
                current_indent + f"// 装载矩阵B第{round_idx}轮的数据: kr={kr}, nr={nr}",
                [],
                [],
            )
        )
        b_row_addr_temp_reg = b_row_addr_temp_regs[
            round_idx % len(b_row_addr_temp_regs)
        ]
        remained_nr = nr
        while remained_nr > 0:
            stored_nr = nr - remained_nr
            c = ceil(stored_nr / elements_per_vector_reg)
            if remained_nr >= (2 * elements_per_vector_reg):
                instructions.append(
                    (
                        [b_row_addr_temp_reg],
                        current_indent
                        + f"ldp {round_scheduler[c].q_reg_name}, {round_scheduler[c + 1].q_reg_name}, [{b_row_addr_temp_reg.reg_name}], #{2 * self._get_vector_byte_width()}   // 装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        [],
                        [round_scheduler[c], round_scheduler[c + 1]],
                    )
                )
                remained_nr -= 2 * elements_per_vector_reg
            elif remained_nr >= elements_per_vector_reg:
                instructions.append(
                    (
                        [b_row_addr_temp_reg],
                        current_indent
                        + f"ldr {round_scheduler[c].q_reg_name}, [{b_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        [],
                        [round_scheduler[c]],
                    )
                )
                remained_nr -= elements_per_vector_reg
            elif remained_nr == 3:
                inst = (
                    current_indent
                    + f"ldr {round_scheduler[c].q_reg_name}, [{b_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 2}列对应的向量寄存器"
                )
                inst += (
                    "\n"
                    + current_indent
                    + f"ins {round_scheduler[c].v_reg_name}.s[3], {round_scheduler[c].v_reg_name}.s[0]    // 用矩阵B第{round_idx}行, 第{c * elements_per_vector_reg}列对应的元素清零向量寄存器的第3个元素, 防止计算出错"
                )
                instructions.append(
                    ([b_row_addr_temp_reg], inst, [], [round_scheduler[c]])
                )
                remained_nr -= 3
            elif remained_nr == 2:
                inst = (
                    current_indent
                    + f"movi {round_scheduler[c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                )
                inst += (
                    "\n"
                    + current_indent
                    + f"ldr {round_scheduler[c].d_reg_name}, [{b_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 - 1}对应的列向量寄存器"
                )
                instructions.append(
                    ([b_row_addr_temp_reg], inst, [], [round_scheduler[c]])
                )
                remained_nr -= 2
            elif remained_nr == 1:
                inst = (
                    current_indent
                    + f"movi {round_scheduler[c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                )
                inst += (
                    "\n"
                    + current_indent
                    + f"ldr {round_scheduler[c].s_reg_name}, [{b_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 - 1}对应的列向量寄存器"
                )
                instructions.append(
                    ([b_row_addr_temp_reg], inst, [], [round_scheduler[c]])
                )
                remained_nr -= 1
            else:
                raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的加载")

            # 考虑预取
            if ((c % 4) == 0) and (
                self.params.b_prefetch != PrefetchCacheLevel.NONE.value
            ):
                instructions.append(
                    (
                        [b_row_addr_temp_reg],
                        current_indent
                        + f"prfm PLD{self.params.b_prefetch.value.upper()}STRM, [{b_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_b.reg_name}]   // 预取B矩阵数据",
                        [],
                        [],
                    )
                )

        # 添加地址计算
        if round_idx + 1 < kr:
            mat_b_row_addr_update_instructions = self._generate_b_row_addr(
                kr, b_row_addr_temp_regs, with_prfm=True, row_idx=round_idx + 1
            )
            instructions.append(
                ([], "\n".join(mat_b_row_addr_update_instructions), [], [])
            )
        return instructions

    def _load_mat_b_data_from_packed_bc_in_kernel_compute(
        self,
        kr: int,
        nr: int,
        mat_b_vec_regs_scheduler: List[List[AllocatedVectorRegister]],
        round_idx: int,
    ) -> List[
        Tuple[
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            str,
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
            List[AllocatedVectorRegister | AllocatedGeneralRegister],
        ]
    ]:
        """
        生成内核计算体中装载矩阵B数据的指令
        """
        if round_idx >= len(mat_b_vec_regs_scheduler):
            raise ValueError(f"round_idx={round_idx} 超出矩阵B向量寄存器调度器范围")
        instructions = []
        current_indent = "    "
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        round_scheduler = mat_b_vec_regs_scheduler[round_idx]
        mat_b_rows, mat_b_cols = kr, len(round_scheduler)
        instructions.append(
            (
                [],
                current_indent
                + f"// 从Bc装载矩阵B第{round_idx}轮的数据: kr={kr}, nr={nr}",
                [],
                [],
            )
        )
        remained_nr = nr
        while remained_nr > 0:
            stored_nr = nr - remained_nr
            c = ceil(stored_nr / elements_per_vector_reg)
            if remained_nr >= (2 * elements_per_vector_reg):
                instructions.append(
                    (
                        [self.register_allocation.pointer_reg_bc_for_loop_k],
                        current_indent
                        + f"ldp {round_scheduler[c].q_reg_name}, {round_scheduler[c + 1].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{2 * self._get_vector_byte_width()}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        [],
                        [round_scheduler[c], round_scheduler[c + 1]],
                    )
                )
                remained_nr -= 2 * elements_per_vector_reg
            elif remained_nr >= elements_per_vector_reg:
                instructions.append(
                    (
                        [self.register_allocation.pointer_reg_bc_for_loop_k],
                        current_indent
                        + f"ldr {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 1}列对应的向量寄存器",
                        [],
                        [round_scheduler[c]],
                    )
                )
                remained_nr -= elements_per_vector_reg
            elif remained_nr == 3:
                inst = (
                    current_indent
                    + f"ldr {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 2}列对应的向量寄存器"
                )
                # inst += (
                #         '\n'
                #         + current_indent + f"ins {round_scheduler[c].v_reg_name}.s[3], {round_scheduler[c].v_reg_name}.s[0]    // 用矩阵B第{round_idx}行, 第{c * elements_per_vector_reg}列对应的元素清零向量寄存器的第3个元素, 防止计算出错"
                # )
                instructions.append(
                    (
                        [self.register_allocation.pointer_reg_bc_for_loop_k],
                        inst,
                        [],
                        [round_scheduler[c]],
                    )
                )
                remained_nr -= 3
            elif remained_nr == 2:
                # inst = current_indent + f"movi {round_scheduler[c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                # inst += (
                #         '\n'
                #         + current_indent + f"ldr {round_scheduler[c].d_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width() // 2}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 - 1}对应的列向量寄存器"
                # )
                inst = (
                    current_indent
                    + f"ldr {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2 - 1}对应的列向量寄存器"
                )
                instructions.append(
                    (
                        [self.register_allocation.pointer_reg_bc_for_loop_k],
                        inst,
                        [],
                        [round_scheduler[c]],
                    )
                )
                remained_nr -= 2
            elif remained_nr == 1:
                # inst = current_indent + f"movi {round_scheduler[c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                # inst += (
                #         '\n'
                #         + current_indent + f"ldr {round_scheduler[c].s_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width() // 4}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg  +  1 - 1}对应的列向量寄存器"
                # )
                inst = (
                    current_indent
                    + f"ldr {round_scheduler[c].q_reg_name}, [{self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}], #{self._get_vector_byte_width()}   // 从Bc装载矩阵B第{round_idx}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 - 1}对应的列向量寄存器"
                )

                instructions.append(
                    (
                        [self.register_allocation.pointer_reg_bc_for_loop_k],
                        inst,
                        [],
                        [round_scheduler[c]],
                    )
                )
                remained_nr -= 1
            else:
                raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的加载")

        return instructions

    def generate_kernel_compute_with_packing(
        self,
        packing_kernel_macro_fmt: Template,
        available_temp_regs: List[AllocatedGeneralRegister],
    ) -> List[Dict[str, List[str] | Any]]:
        """
        生成内核计算体
        """
        kernel_compute_macros = list()
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        kernel_shapes = self.generate_kernel_shapes()
        for mr, kr, nr in kernel_shapes:
            available_temp_regs_for_loop = available_temp_regs.copy()
            kernel_instructions = [
                f"/**** 打包内核说明",
                current_indent + f"* 内核大小: mr={mr}, kr={kr}, nr={nr}",
                current_indent
                + f"* bc地址寄存器: {self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}",
            ]
            a_row_addr_temp_regs = [
                available_temp_regs_for_loop.pop(0) for _ in range(mr)
            ]
            kernel_instructions.extend(
                [
                    current_indent
                    + f"* 矩阵A第{i}行的地址寄存器: {a_row_addr_temp_regs[i].reg_name}"
                    for i in range(mr)
                ]
            )
            mat_a_vec_regs_scheduler, mat_b_vec_regs_scheduler = (
                self._get_vector_reg_scheduler_for_kernel_compute(mr, kr, nr)
            )
            kernel_instructions.append(
                current_indent
                + f"* 矩阵A实际使用的向量寄存器: {','.join([reg.v_reg_name for reg in sorted(list(set(chain.from_iterable(mat_a_vec_regs_scheduler))), key=lambda x: x.reg_id)])}"
            )
            kernel_instructions.append(current_indent + "* 矩阵A向量寄存器调度表:")
            for r in range(len(mat_a_vec_regs_scheduler)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}轮: "
                    + ",".join([reg.v_reg_name for reg in mat_a_vec_regs_scheduler[r]])
                )
            b_row_addr_temp_regs = [
                available_temp_regs_for_loop.pop(0) for _ in range(2)
            ]  # 矩阵B至少需要2个临时寄存器
            kernel_instructions.extend(
                [
                    current_indent
                    + f"* 矩阵B第{j}行的地址寄存器: {b_row_addr_temp_regs[j].reg_name}"
                    for j in range(2)
                ]
            )

            kernel_instructions.append(
                current_indent
                + f"* 矩阵B实际使用的向量寄存器: {','.join([reg.v_reg_name for reg in sorted(list(set(chain.from_iterable(mat_b_vec_regs_scheduler))), key=lambda x: x.reg_id)])}"
            )
            kernel_instructions.append(current_indent + "* 矩阵B向量寄存器调度表:")
            for r in range(len(mat_b_vec_regs_scheduler)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}轮: "
                    + ",".join([reg.v_reg_name for reg in mat_b_vec_regs_scheduler[r]])
                )

            kernel_instructions.append("* 矩阵C使用的寄存器列表: ")
            mat_c_vec_regs, _ = (
                self.register_allocation.c_vector_regs.get_rearranged_reg_names(
                    mr, ceil(nr / elements_per_vector_reg)
                )
            )
            for r in range(len(mat_c_vec_regs)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}行: "
                    + ",".join([reg.v_reg_name for reg in mat_c_vec_regs[r]])
                )

            remained_k_reg = available_temp_regs_for_loop.pop(0)
            kernel_instructions.append(
                current_indent
                + f"* 计算剩余K维度的临时寄存器: {remained_k_reg.reg_name}"
            )
            kernel_instructions.append("****/")

            kernel_instructions.extend(
                self._generate_a_row_addr(mr, a_row_addr_temp_regs, True)
            )
            kernel_instructions.extend(
                self._generate_b_row_addr(kr, b_row_addr_temp_regs, True)
            )

            kernel_instructions.append(
                current_indent
                + f"sub {remained_k_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 初始化剩余K维度计数器"
            )

            kernel_instructions.append(
                f"PACKING_KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_INIT_{label_suffix} :    // 初始化内核计算"
            )
            # 装载矩阵A的首轮数据
            mat_a_first_loop_instructions = (
                self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                    mr, kr, a_row_addr_temp_regs, mat_a_vec_regs_scheduler, round_idx=0
                )
            )
            kernel_instructions.extend(
                [inst[1] for inst in mat_a_first_loop_instructions]
            )
            # 装载矩阵B的首轮数据
            mat_b_first_loop_instructions = self._load_mat_b_data_in_kernel_compute(
                kr, nr, b_row_addr_temp_regs, mat_b_vec_regs_scheduler, round_idx=0
            )
            kernel_instructions.extend(
                [inst[1] for inst in mat_b_first_loop_instructions]
            )

            # 判断是否进入ping-pong计算
            kernel_instructions.append(
                current_indent + f"// 判断是否进入Ping-Pong计算: kr={kr}"
            )
            kernel_instructions.append(
                current_indent
                + f"cmp {remained_k_reg.reg_name}, #{2 * kr}   // 判断是否进入Ping-Pong计算: kr={kr}"
            )
            kernel_instructions.append(
                current_indent
                + f"blt PACKING_KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_FINISH_{label_suffix}    // 如果剩余K维度小于2*kr={2 * kr}，跳转到最后一轮计算"
            )

            # 更新初期可用寄存器集合
            init_available_regs_set = set()
            init_available_regs_set.add(
                self.register_allocation.pointer_reg_bc_for_loop_k
            )
            init_available_regs_set.update(a_row_addr_temp_regs)
            init_available_regs_set.update(b_row_addr_temp_regs)
            for r in range(len(mat_c_vec_regs)):
                init_available_regs_set.update(mat_c_vec_regs[r])

            for inst in mat_a_first_loop_instructions:
                init_available_regs_set.difference_update(inst[2])
                init_available_regs_set.update(inst[3])

            for inst in mat_b_first_loop_instructions:
                init_available_regs_set.difference_update(inst[2])
                init_available_regs_set.update(inst[3])

            available_regs_set = deepcopy(init_available_regs_set)
            # Ping-Pong计算
            kernel_instructions.append(
                f"PACKING_KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_PINGPONG_{label_suffix} :    // 内核计算体Ping-Pong计算开始, mr={mr}, kr={kr}, nr={nr}"
            )
            compute_round_list = [i for i in range(0, kr)]
            load_round_list = compute_round_list[1:] + compute_round_list[:1]
            for compute_round_idx, load_round_idx in zip(
                compute_round_list, load_round_list
            ):
                if load_round_idx == 0:
                    kernel_instructions.extend(self._update_dealt_k_for_dim_k(kr))
                    kernel_instructions.extend(
                        self._generate_b_row_addr(
                            kr, b_row_addr_temp_regs, with_prfm=True, row_idx=None
                        )
                    )
                # 计算结果
                compute_instructions = self._compute_result_in_kernel_compute(
                    mr,
                    kr,
                    nr,
                    mat_a_vec_regs_scheduler,
                    mat_b_vec_regs_scheduler,
                    compute_round_idx,
                )
                # 打包矩阵B数据
                packing_mat_b_instructions = self._packing_mat_b_data_in_kernel_compute(
                    kr, nr, mat_b_vec_regs_scheduler, compute_round_idx
                )
                # 装载矩阵A数据
                mat_a_load_round, mat_a_load_trigger = divmod(
                    load_round_idx, elements_per_vector_reg
                )
                if mat_a_load_trigger == 0:
                    load_mat_a_instructions = (
                        self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                            mr,
                            kr,
                            a_row_addr_temp_regs,
                            mat_a_vec_regs_scheduler,
                            mat_a_load_round,
                        )
                    )
                else:
                    load_mat_a_instructions = []
                # 装载矩阵B数据
                load_mat_b_instructions = self._load_mat_b_data_in_kernel_compute(
                    kr,
                    nr,
                    b_row_addr_temp_regs,
                    mat_b_vec_regs_scheduler,
                    load_round_idx,
                )

                scheduled_insts = self._schedule_inst_seq(  # 进行调度
                    inst_seqs=[
                        compute_instructions,
                        packing_mat_b_instructions,
                        load_mat_a_instructions,
                        load_mat_b_instructions,
                    ],
                    available_regs_set=available_regs_set,
                )
                kernel_instructions.extend(scheduled_insts)

                if compute_round_idx == (kr - 1):
                    # 更新剩余K维度计数器
                    kernel_instructions.append(
                        current_indent
                        + f"sub {remained_k_reg.reg_name}, {remained_k_reg.reg_name}, #{kr}   // 更新剩余K维度计数器"
                    )
                    # 判断是否继续Ping-Pong计算
                    kernel_instructions.append(
                        current_indent
                        + f"cmp {remained_k_reg.reg_name}, #{2 * kr}   // 判断是否继续Ping-Pong计算: kr={kr}"
                    )
                    kernel_instructions.append(
                        current_indent
                        + f"bge PACKING_KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_PINGPONG_{label_suffix}    // 如果剩余K维度大于等于2*kr={2 * kr}，继续Ping-Pong计算"
                    )

            # 最后一轮计算
            available_regs_set = deepcopy(init_available_regs_set)
            kernel_instructions.append(
                f"PACKING_KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_FINISH_{label_suffix} :    // 最后一轮计算开始, mr={mr}, kr={kr}, nr={nr}"
            )
            for compute_round_idx, load_round_idx in zip(
                compute_round_list, load_round_list
            ):
                # 计算结果
                compute_instructions = self._compute_result_in_kernel_compute(
                    mr,
                    kr,
                    nr,
                    mat_a_vec_regs_scheduler,
                    mat_b_vec_regs_scheduler,
                    compute_round_idx,
                )
                # 打包矩阵B数据
                packing_mat_b_instructions = self._packing_mat_b_data_in_kernel_compute(
                    kr, nr, mat_b_vec_regs_scheduler, compute_round_idx
                )
                if load_round_idx > 0:
                    # 装载矩阵A数据
                    mat_a_load_round, mat_a_load_trigger = divmod(
                        load_round_idx, elements_per_vector_reg
                    )
                    if mat_a_load_trigger == 0:
                        load_mat_a_instructions = (
                            self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                                mr,
                                kr,
                                a_row_addr_temp_regs,
                                mat_a_vec_regs_scheduler,
                                mat_a_load_round,
                            )
                        )
                    else:
                        load_mat_a_instructions = []
                    # 装载矩阵B数据
                    load_mat_b_instructions = self._load_mat_b_data_in_kernel_compute(
                        kr,
                        nr,
                        b_row_addr_temp_regs,
                        mat_b_vec_regs_scheduler,
                        load_round_idx,
                    )
                else:
                    load_mat_a_instructions = []
                    load_mat_b_instructions = []

                scheduled_insts = self._schedule_inst_seq(  # 进行调度
                    inst_seqs=[
                        compute_instructions,
                        packing_mat_b_instructions,
                        load_mat_a_instructions,
                        load_mat_b_instructions,
                    ],
                    available_regs_set=available_regs_set,
                )
                kernel_instructions.extend(scheduled_insts)

                if load_round_idx == 0:
                    kernel_instructions.extend(self._update_dealt_k_for_dim_k(kr))

            kernel_compute_macros.append(
                {
                    "name": packing_kernel_macro_fmt.substitute(mr=mr, kr=kr, nr=nr),
                    "instructions": kernel_instructions,
                }
            )

        return kernel_compute_macros

    def generate_kernel_compute_use_packing(
        self,
        kernel_macro_fmt: Template,
        available_temp_regs: List[AllocatedGeneralRegister],
    ) -> List[Dict[str, List[str] | Any]]:
        """
        生成内核计算体
        """
        kernel_compute_macros = list()
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        kernel_shapes = self.generate_kernel_shapes()
        for mr, kr, nr in kernel_shapes:
            available_temp_regs_for_loop = available_temp_regs.copy()
            kernel_instructions = [
                f"/**** 内核说明(矩阵B使用打包后的数据)",
                current_indent + f"* 内核大小: mr={mr}, kr={kr}, nr={nr}",
                current_indent
                + f"* 打包后的bc地址寄存器: {self.register_allocation.pointer_reg_bc_for_loop_k.reg_name}",
            ]
            a_row_addr_temp_regs = [
                available_temp_regs_for_loop.pop(0) for _ in range(mr)
            ]
            kernel_instructions.extend(
                [
                    current_indent
                    + f"* 矩阵A第{i}行的地址寄存器: {a_row_addr_temp_regs[i].reg_name}"
                    for i in range(mr)
                ]
            )
            mat_a_vec_regs_scheduler, mat_b_vec_regs_scheduler = (
                self._get_vector_reg_scheduler_for_kernel_compute(mr, kr, nr)
            )
            kernel_instructions.append(
                current_indent
                + f"* 矩阵A实际使用的向量寄存器: {','.join([reg.v_reg_name for reg in sorted(list(set(chain.from_iterable(mat_a_vec_regs_scheduler))), key=lambda x: x.reg_id)])}"
            )
            kernel_instructions.append(current_indent + "* 矩阵A向量寄存器调度表:")
            for r in range(len(mat_a_vec_regs_scheduler)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}轮: "
                    + ",".join([reg.v_reg_name for reg in mat_a_vec_regs_scheduler[r]])
                )

            kernel_instructions.append(
                current_indent
                + f"* 矩阵B实际使用的向量寄存器: {','.join([reg.v_reg_name for reg in sorted(list(set(chain.from_iterable(mat_b_vec_regs_scheduler))), key=lambda x: x.reg_id)])}"
            )
            kernel_instructions.append(current_indent + "* 矩阵B向量寄存器调度表:")
            for r in range(len(mat_b_vec_regs_scheduler)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}轮: "
                    + ",".join([reg.v_reg_name for reg in mat_b_vec_regs_scheduler[r]])
                )

            kernel_instructions.append("* 矩阵C使用的寄存器列表: ")
            mat_c_vec_regs, _ = (
                self.register_allocation.c_vector_regs.get_rearranged_reg_names(
                    mr, ceil(nr / elements_per_vector_reg)
                )
            )
            for r in range(len(mat_c_vec_regs)):
                kernel_instructions.append(
                    current_indent
                    + f"* 第{r}行: "
                    + ",".join([reg.v_reg_name for reg in mat_c_vec_regs[r]])
                )

            remained_k_reg = available_temp_regs_for_loop.pop(0)
            kernel_instructions.append(
                current_indent
                + f"* 计算剩余K维度的临时寄存器: {remained_k_reg.reg_name}"
            )
            kernel_instructions.append("****/")

            kernel_instructions.extend(
                self._generate_a_row_addr(mr, a_row_addr_temp_regs, True)
            )

            kernel_instructions.append(
                current_indent
                + f"sub {remained_k_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 初始化剩余K维度计数器"
            )

            kernel_instructions.append(
                f"KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_INIT_{label_suffix} :    // 初始化内核计算"
            )
            # 装载矩阵A的首轮数据
            mat_a_first_loop_instructions = (
                self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                    mr, kr, a_row_addr_temp_regs, mat_a_vec_regs_scheduler, round_idx=0
                )
            )
            kernel_instructions.extend(
                [inst[1] for inst in mat_a_first_loop_instructions]
            )
            # 装载矩阵B的首轮数据
            mat_b_first_loop_instructions = (
                self._load_mat_b_data_from_packed_bc_in_kernel_compute(
                    kr, nr, mat_b_vec_regs_scheduler, round_idx=0
                )
            )
            kernel_instructions.extend(
                [inst[1] for inst in mat_b_first_loop_instructions]
            )

            # 判断是否进入ping-pong计算
            kernel_instructions.append(
                current_indent + f"// 判断是否进入Ping-Pong计算: kr={kr}"
            )
            kernel_instructions.append(
                current_indent
                + f"cmp {remained_k_reg.reg_name}, #{2 * kr}   // 判断是否进入Ping-Pong计算: kr={kr}"
            )
            kernel_instructions.append(
                current_indent
                + f"blt KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_FINISH_{label_suffix}    // 如果剩余K维度小于2*kr={2 * kr}，跳转到最后一轮计算"
            )

            # 更新初期可用寄存器集合
            init_available_regs_set = set()
            init_available_regs_set.add(
                self.register_allocation.pointer_reg_bc_for_loop_k
            )
            init_available_regs_set.update(a_row_addr_temp_regs)
            for r in range(len(mat_c_vec_regs)):
                init_available_regs_set.update(mat_c_vec_regs[r])

            for inst in mat_a_first_loop_instructions:
                init_available_regs_set.difference_update(inst[2])
                init_available_regs_set.update(inst[3])

            for inst in mat_b_first_loop_instructions:
                init_available_regs_set.difference_update(inst[2])
                init_available_regs_set.update(inst[3])

            available_regs_set = deepcopy(init_available_regs_set)
            # Ping-Pong计算
            kernel_instructions.append(
                f"KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_PINGPONG_{label_suffix} :    // 内核计算体Ping-Pong计算开始, mr={mr}, kr={kr}, nr={nr}"
            )
            compute_round_list = [i for i in range(0, kr)]
            load_round_list = compute_round_list[1:] + compute_round_list[:1]
            for compute_round_idx, load_round_idx in zip(
                compute_round_list, load_round_list
            ):
                if load_round_idx == 0:
                    kernel_instructions.extend(self._update_dealt_k_for_dim_k(kr))
                # 计算结果
                compute_instructions = self._compute_result_in_kernel_compute(
                    mr,
                    kr,
                    nr,
                    mat_a_vec_regs_scheduler,
                    mat_b_vec_regs_scheduler,
                    compute_round_idx,
                )
                # 装载矩阵A数据
                mat_a_load_round, mat_a_load_trigger = divmod(
                    load_round_idx, elements_per_vector_reg
                )
                if mat_a_load_trigger == 0:
                    load_mat_a_instructions = (
                        self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                            mr,
                            kr,
                            a_row_addr_temp_regs,
                            mat_a_vec_regs_scheduler,
                            mat_a_load_round,
                        )
                    )
                else:
                    load_mat_a_instructions = []
                # 装载矩阵B数据
                load_mat_b_instructions = (
                    self._load_mat_b_data_from_packed_bc_in_kernel_compute(
                        kr, nr, mat_b_vec_regs_scheduler, load_round_idx
                    )
                )

                scheduled_insts = self._schedule_inst_seq(  # 进行调度
                    inst_seqs=[
                        compute_instructions,
                        load_mat_a_instructions,
                        load_mat_b_instructions,
                    ],
                    available_regs_set=available_regs_set,
                )
                kernel_instructions.extend(scheduled_insts)

                if compute_round_idx == (kr - 1):
                    # 更新剩余K维度计数器
                    kernel_instructions.append(
                        current_indent
                        + f"sub {remained_k_reg.reg_name}, {remained_k_reg.reg_name}, #{kr}   // 更新剩余K维度计数器"
                    )
                    # 判断是否继续Ping-Pong计算
                    kernel_instructions.append(
                        current_indent
                        + f"cmp {remained_k_reg.reg_name}, #{2 * kr}   // 判断是否继续Ping-Pong计算: kr={kr}"
                    )
                    kernel_instructions.append(
                        current_indent
                        + f"bge KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_PINGPONG_{label_suffix}    // 如果剩余K维度大于等于2*kr={2 * kr}，继续Ping-Pong计算"
                    )

            # 最后一轮计算
            available_regs_set = deepcopy(init_available_regs_set)
            kernel_instructions.append(
                f"KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_FINISH_{label_suffix} :    // 最后一轮计算开始, mr={mr}, kr={kr}, nr={nr}"
            )
            for compute_round_idx, load_round_idx in zip(
                compute_round_list, load_round_list
            ):
                # 计算结果
                compute_instructions = self._compute_result_in_kernel_compute(
                    mr,
                    kr,
                    nr,
                    mat_a_vec_regs_scheduler,
                    mat_b_vec_regs_scheduler,
                    compute_round_idx,
                )
                if load_round_idx > 0:
                    # 装载矩阵A数据
                    mat_a_load_round, mat_a_load_trigger = divmod(
                        load_round_idx, elements_per_vector_reg
                    )
                    if mat_a_load_trigger == 0:
                        load_mat_a_instructions = (
                            self._load_mat_a_data_in_kernel_compute_with_alpha_scale(
                                mr,
                                kr,
                                a_row_addr_temp_regs,
                                mat_a_vec_regs_scheduler,
                                mat_a_load_round,
                            )
                        )
                    else:
                        load_mat_a_instructions = []
                    # 装载矩阵B数据
                    load_mat_b_instructions = (
                        self._load_mat_b_data_from_packed_bc_in_kernel_compute(
                            kr, nr, mat_b_vec_regs_scheduler, load_round_idx
                        )
                    )
                else:
                    load_mat_a_instructions = []
                    load_mat_b_instructions = []

                scheduled_insts = self._schedule_inst_seq(  # 进行调度
                    inst_seqs=[
                        compute_instructions,
                        load_mat_a_instructions,
                        load_mat_b_instructions,
                    ],
                    available_regs_set=available_regs_set,
                )
                kernel_instructions.extend(scheduled_insts)

                if load_round_idx == 0:
                    kernel_instructions.extend(self._update_dealt_k_for_dim_k(kr))

            kernel_compute_macros.append(
                {
                    "name": kernel_macro_fmt.substitute(mr=mr, kr=kr, nr=nr),
                    "instructions": kernel_instructions,
                }
            )

        return kernel_compute_macros

    def generate_main_loop_logic(self) -> Dict[str, List[str] | Any]:
        """生成主计算循环"""
        self.register_allocation.clear()
        main_loop_n_instructions = []
        current_indent = "    "
        label_suffix = self.get_label_suffix()

        main_loop_n_instructions.extend(
            [
                # current_indent + f"eor {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}  // 清零K维度计数器",
                current_indent
                + f"eor {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}  // 清零N维度计数器",
                current_indent
                + f"lsl {self.register_allocation.stride_reg_lda.reg_name}, {self.register_allocation.stride_reg_lda.reg_name}, #{self.params.get_dtype_lsl_position()}    // lda*{1 << self.params.get_dtype_lsl_position()}转换为字节偏移",
                current_indent
                + f"lsl {self.register_allocation.stride_reg_ldb.reg_name}, {self.register_allocation.stride_reg_ldb.reg_name}, #{self.params.get_dtype_lsl_position()}    // ldb*{1 << self.params.get_dtype_lsl_position()}转换为字节偏移",
                current_indent
                + f"lsl {self.register_allocation.stride_reg_ldc.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}, #{self.params.get_dtype_lsl_position()}    // ldc*{1 << self.params.get_dtype_lsl_position()}转换为字节偏移",
            ]
        )
        nr_list = self.params.get_nr_list()
        nr_list.append(0)  # 方便生成最后一个循环
        self.register_allocation.temp_regs.set_infinite_iteration(True)
        n_temp_reg = next(self.register_allocation.temp_regs)
        m_temp_reg = next(self.register_allocation.temp_regs)
        k_temp_reg = next(self.register_allocation.temp_regs)
        loop_m_macro_fmt = Template("LOOP_NR${nr}_MR" + f"_{label_suffix}")
        main_loop_n_instructions.extend(
            self.generate_loop_n(n_temp_reg, loop_m_macro_fmt)
        )  # 生成N循环体
        main_loop_n_instructions.append(
            f"MAIN_LOOP_END_{label_suffix}:    // N维度(主)循环结束标记"
        )

        # 处理M维度的循环
        loop_k_macro_fmt = Template(
            "LOOP_NR${nr}_MR${mr}_KR" + f"_{label_suffix}"
        )  # K维度主力核
        loop_k_packing_macro_fmt = Template(
            "LOOP_NR${nr}_MR${mr}_KR_PACKING" + f"_{label_suffix}"
        )  # K维度主力核
        main_loop_m_instructions = self.generate_loop_m_with_packing(
            m_temp_reg, loop_m_macro_fmt, loop_k_macro_fmt, loop_k_packing_macro_fmt
        )  # 生成M循环体

        # 处理K维度的循环
        store_c_macro_fmt = Template("STORE_C_NR${nr}_MR${mr}" + f"_{label_suffix}")
        load_c_macro_fmt = Template("LOAD_C_NR${nr}_MR${mr}" + f"_{label_suffix}")
        main_loop_k_instructions = self.generate_loop_k_use_packing(
            k_temp_reg,
            loop_k_macro_fmt,
            self.get_kernel_macro_fmt(),
            load_c_macro_fmt,
            store_c_macro_fmt,
        )  # 生成主力K循环体
        main_loop_k_instructions.extend(
            self.generate_loop_k_with_packing(
                k_temp_reg,
                loop_k_packing_macro_fmt,
                self.get_packing_kernel_macro_fmt(),
                load_c_macro_fmt,
                store_c_macro_fmt,
            )
        )  # 生成打包K循环体

        c_row_addr_temp_regs = [
            next(self.register_allocation.temp_regs) for _ in range(self.params.mr)
        ]
        load_c_instructions = self.generate_load_c(
            load_c_macro_fmt, c_row_addr_temp_regs
        )
        store_c_instructions = self.generate_store_c(
            store_c_macro_fmt, c_row_addr_temp_regs
        )

        available_temp_regs = [
            next(self.register_allocation.temp_regs)
            for _ in range(
                self.params.mr + 3
            )  # 用于地址计算: 矩阵A至少需要mr个, 矩阵B至少需要2个
        ]
        kernel_compute_instructions = self.generate_kernel_compute_with_packing(
            self.get_packing_kernel_macro_fmt(), available_temp_regs
        )  # 生成打包内核
        kernel_compute_instructions.extend(
            self.generate_kernel_compute_use_packing(
                self.get_kernel_macro_fmt(), available_temp_regs
            )
        )  # 生成利用打包数据的内核

        return {
            "main_loop_n": main_loop_n_instructions,
            "main_loop_m": main_loop_m_instructions,
            "main_loop_k": main_loop_k_instructions,
            "load_c": load_c_instructions,
            "store_c": store_c_instructions,
            "kernel_compute": kernel_compute_instructions,
        }
