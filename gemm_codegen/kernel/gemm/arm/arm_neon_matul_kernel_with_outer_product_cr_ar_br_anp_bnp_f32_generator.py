# -*- coding: utf-8 -*-
# Author        : huche
# Create Time   : 2025/12/11 16:34
# Contact       : hucheng.lew@qq.com
# File          : arm_neon_matul_kernel_with_outer_product_cr_ar_br_f32_generator.py
# IDE           : PyCharm
# Description   : 采用外积法实现的 ARM NEON 矩阵乘法内核生成器（C行主、A行主、B行主）
import json
from copy import deepcopy
from itertools import chain, pairwise, zip_longest
from math import ceil, floor
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from scipy.constants import precision

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

from .arm_neon_matmul_kernel_with_outer_product_base_generator import (
    ArmNeonMatMulKernelWithOuterProductBaseGenerator,
)


class ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator(
    ArmNeonMatMulKernelWithOuterProductBaseGenerator
):
    """
    采用外积法实现的 ARM NEON 矩阵乘法内核生成器（C行主、A行主、B行主, AB均无打包）
    """

    def __init__(self, params: KernelParameters):
        super().__init__(params)

        if self.params.data_type.precision.value != PrecisionType.FP32.value:
            raise ValueError(
                "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator 仅支持 f32 数据类型"
            )

        if not (
            (self.params.c_layout.value == MatrixLayout.ROW_MAJOR.value)
            and (self.params.a_layout.value == MatrixLayout.ROW_MAJOR.value)
            and (self.params.b_layout.value == MatrixLayout.ROW_MAJOR.value)
        ):
            raise ValueError(
                "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator 仅支持 C、A、B 均为行主序布局"
            )

        if type(self) is ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator:
            if (self.params.a_packing.value != PackingStrategy.NO_PACKING.value) or (
                self.params.b_packing.value != PackingStrategy.NO_PACKING.value
            ):
                raise ValueError(
                    "ArmNeonMatMulKernelWithOuterProductCrArBrAnpBnpF32Generator 仅支持 A、B 均无打包"
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
        """
        生成函数序言
        """

        instructions = super().generate_function_prologue()
        instructions.extend(
            [
                f"mov {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}], v0.s[0]         // 将alpha的值复制到{self.register_allocation.scalar_reg_alpha.v_reg_name}",
                f"mov {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}], v1.s[0]           // 将beta的值复制到{self.register_allocation.scalar_reg_beta.v_reg_name}",
            ]
        )
        return instructions

    def generate_kernel_shapes(self) -> List[Tuple[int, int, int]]:
        """生成边界内核形状列表"""
        kernel_shapes = set()
        mr, nr, kr = self.params.mr, self.params.nr, self.params.kr
        elements_per_vector_reg = self.get_elements_per_vector_reg()

        if self.params.support_edge_opt:
            # 计算nr的边界
            if nr > elements_per_vector_reg:
                n_list = [1, 2]
                n_list.extend(
                    [
                        i
                        for i in range(
                            elements_per_vector_reg, nr + 1, elements_per_vector_reg
                        )
                    ]
                )
            elif nr == elements_per_vector_reg:
                n_list = [1, 2, elements_per_vector_reg]
            else:
                n_list = [1, 2]
            # 计算kr的边界
            if kr > elements_per_vector_reg:
                k_list = [1, 2]
                k_list.extend(
                    [
                        i
                        for i in range(
                            elements_per_vector_reg, kr + 1, elements_per_vector_reg
                        )
                    ]
                )
            elif kr == elements_per_vector_reg:
                k_list = [1, 2, elements_per_vector_reg]
            else:
                k_list = [1, 2]

            for m in range(1, mr + 1):
                for n in n_list:
                    for k in k_list:
                        kernel_shapes.add((m, k, n))
        else:
            kernel_shapes.add((mr, kr, nr))
        return sorted(
            list(kernel_shapes), key=lambda x: (x[0], x[2], x[1]), reverse=True
        )

    def generate_store_c(
        self,
        store_c_macro_fmt: Template,
        c_row_addr_temp_regs: List[AllocatedGeneralRegister],
    ) -> List[Dict[str, List[str] | Any]]:
        """
        生成保存C的指令
        """
        store_c_macros = list()
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        nr_list = self.params.get_nr_list()
        for nr in nr_list:
            mr_list = self.params.get_mr_list(nr)
            for mr in mr_list:
                if len(c_row_addr_temp_regs) < mr:
                    raise NotImplementedError(
                        f"当前不支持mr={mr}大于临时寄存器数量的情况，请增加临时寄存器数量"
                    )
                rows, cols = mr, ceil(nr / vector_reg_element_width)
                used_vec_regs, idle_vec_regs = (
                    self.register_allocation.c_vector_regs.get_rearranged_reg_names(
                        rows, cols
                    )
                )
                store_instructions = []
                store_instructions.append(
                    current_indent + f"// 保存C矩阵块: mr={mr}, nr={nr}"
                )

                store_instructions.append(current_indent + "// 计算每行的地址")
                for r in range(rows):
                    store_instructions.append(
                        current_indent + f"// 计算C矩阵第{r}行地址并保存"
                    )
                    c_row_addr_temp_reg = c_row_addr_temp_regs[r]
                    if r == 0:
                        store_instructions.append(
                            current_indent
                            + f"mov {c_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}   // 计算C矩阵第{r}行地址"
                        )
                    else:
                        last_c_row_addr_temp_reg = c_row_addr_temp_regs[r - 1]
                        store_instructions.append(
                            current_indent
                            + f"add {c_row_addr_temp_reg.reg_name}, {last_c_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}   // 计算C矩阵第{r}行地址"
                        )
                store_instructions.append(
                    current_indent + "// 开始保存矩阵C块的数据到内存"
                )
                for r in range(rows):
                    c_row_addr_temp_reg = c_row_addr_temp_regs[r]
                    remained_nr = nr
                    while remained_nr > 0:
                        stored_nr = nr - remained_nr
                        c = ceil(stored_nr / vector_reg_element_width)
                        # 计算剩余的元素数
                        if remained_nr >= 2 * vector_reg_element_width:
                            store_instructions.append(
                                current_indent
                                + f"stp {used_vec_regs[r][c].q_reg_name}, {used_vec_regs[r][c + 1].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{2 * self._get_vector_byte_width()}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 * vector_reg_element_width - 1}列对应的向量寄存器"
                            )
                            remained_nr -= 2 * vector_reg_element_width
                        elif remained_nr >= vector_reg_element_width:
                            store_instructions.append(
                                current_indent
                                + f"str {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + vector_reg_element_width - 1}列对应的向量寄存器"
                            )
                            remained_nr -= vector_reg_element_width
                        elif remained_nr == 3:
                            inst = (
                                current_indent
                                + f"str {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}列对应的向量寄存器"
                            )
                            inst += (
                                "\n"
                                + current_indent
                                + f"ins {used_vec_regs[r][c].v_reg_name}.s[0], {used_vec_regs[r][c].v_reg_name}.s[2]   // 将C矩阵块第{r}行, 第{c * vector_reg_element_width + 3}列对应的数据往前挪为后续的存储做准备"
                            )
                            inst += (
                                "\n"
                                + current_indent
                                + f"str {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width + 3}列对应的向量寄存器"
                            )
                            store_instructions.append(inst)
                            remained_nr -= 3
                        elif remained_nr == 2:
                            store_instructions.append(
                                current_indent
                                + f"str {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}列对应的向量寄存器"
                            )
                            remained_nr -= 2
                        elif remained_nr == 1:
                            store_instructions.append(
                                current_indent
                                + f"str {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 保存C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 - 1}列对应的向量寄存器"
                            )
                            remained_nr -= 1
                        else:
                            raise NotImplementedError(
                                f"不支持的剩余nr={remained_nr}元素数的保存"
                            )

                store_c_macros.append(
                    {
                        "name": store_c_macro_fmt.substitute(nr=nr, mr=mr),
                        "instructions": store_instructions,
                    }
                )
        return store_c_macros

    def __generate_c_row_addr(
        self, mr: int, nr: int, c_row_addr_temp_regs: List[AllocatedGeneralRegister]
    ) -> List[str]:
        """
        生成计算C矩阵行地址的指令
        """
        instructions = []
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        rows, cols = mr, ceil(nr / vector_reg_element_width)
        instructions.append(current_indent + f"// 计算每行的地址: mr={mr}, nr={nr}")
        for r in range(rows):
            instructions.append(current_indent + f"// 计算C矩阵第{r}行地址并装载")
            c_row_addr_temp_reg = c_row_addr_temp_regs[r]
            if r == 0:
                instructions.append(
                    current_indent
                    + f"mov {c_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}   // 计算C矩阵第{r}行地址"
                )
            else:
                last_c_row_addr_temp_reg = c_row_addr_temp_regs[r - 1]
                instructions.append(
                    current_indent
                    + f"add {c_row_addr_temp_reg.reg_name}, {last_c_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}   // 计算C矩阵第{r}行地址"
                )
        return instructions

    def __generate_load_c_with_beta_scale(
        self,
        mr: int,
        nr: int,
        c_row_addr_temp_regs: List[AllocatedGeneralRegister],
        only_support_beta_one: bool,
    ) -> List[str]:
        """
        生成装载C矩阵块并乘以beta的指令
        """
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        rows, cols = mr, ceil(nr / vector_reg_element_width)
        used_vec_regs, idle_vec_regs = (
            self.register_allocation.c_vector_regs.get_rearranged_reg_names(rows, cols)
        )
        load_instructions = [
            current_indent + f"// 开始装载矩阵C块的数据到向量寄存器: mr={mr}, nr={nr}"
        ]
        for r in range(rows):
            c_row_addr_temp_reg = c_row_addr_temp_regs[r]
            remained_nr = nr
            while remained_nr > 0:
                stored_nr = nr - remained_nr
                c = ceil(stored_nr / vector_reg_element_width)
                if remained_nr >= 2 * vector_reg_element_width:
                    inst = (
                        current_indent
                        + f"ldp {used_vec_regs[r][c].q_reg_name}, {used_vec_regs[r][c + 1].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{2 * self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 2) * vector_reg_element_width - 1}列对应的向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                            "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}列对应的向量寄存器乘以beta"
                            + "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c + 1].v_reg_name}.4s, {used_vec_regs[r][c + 1].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]         // C矩阵块第{r}行, 第{(c + 1) * vector_reg_element_width} ~ {(c + 2) * vector_reg_element_width - 1}列对应有的向量寄存器乘以beta"
                        )
                    remained_nr -= 2 * vector_reg_element_width
                elif remained_nr >= vector_reg_element_width:
                    inst = (
                        current_indent
                        + f"ldr {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 * vector_reg_element_width - 1}列对应的向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                            "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= vector_reg_element_width
                elif remained_nr == 3:
                    inst = (
                        current_indent
                        + f"ldr {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 * vector_reg_element_width - 2}列对应的向量寄存器"
                    )
                    inst += (
                        "\n"
                        + current_indent
                        + f"ins {used_vec_regs[r][c].v_reg_name}.s[3], {used_vec_regs[r][c].v_reg_name}.s[0]   // 用C矩阵块第{r}行, 第{c * vector_reg_element_width}列的元素填充一下，避免计算出错"
                    )
                    if not only_support_beta_one:
                        inst += (
                            "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 3
                elif remained_nr == 2:
                    inst = (
                        current_indent
                        + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    )
                    inst += (
                        "\n"
                        + current_indent
                        + f"ldr {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}对应的列向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                            "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 2
                elif remained_nr == 1:
                    inst = (
                        current_indent
                        + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    )
                    inst += (
                        "\n"
                        + current_indent
                        + f"ldr {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 - 1}对应的列向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                            "\n"
                            + current_indent
                            + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 1
                else:
                    raise NotImplementedError(
                        f"不支持的剩余nr={remained_nr}元素数的加载"
                    )
                load_instructions.append(inst)
                if (c % 4 == 0) and (
                    self.params.c_prefetch != PrefetchCacheLevel.NONE.value
                ):
                    load_instructions.append(
                        current_indent
                        + f"prfm PLD{self.params.c_prefetch.value.upper()}STRM, [{c_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_c.reg_name}]   // 预取C矩阵数据"
                    )
        return load_instructions

    def __generate_load_c_with_clear(self, mr: int, nr: int) -> List[str]:
        """
        生成装载C矩阵块并清零的指令
        """
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        rows, cols = mr, ceil(nr / vector_reg_element_width)
        used_vec_regs, idle_vec_regs = (
            self.register_allocation.c_vector_regs.get_rearranged_reg_names(rows, cols)
        )
        load_instructions = [
            current_indent + f"// 开始清空矩阵C的向量寄存器: mr={mr}, nr={nr}"
        ]
        for r in range(rows):
            for c in range(cols):
                load_instructions.append(
                    current_indent
                    + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 清零C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}列对应的向量寄存器"
                )
        return load_instructions

    def generate_load_c(
        self,
        load_c_macro_fmt: Template,
        c_row_addr_temp_regs: List[AllocatedGeneralRegister],
    ) -> List[Dict[str, List[str] | Any]]:
        """
        生成装载C的指令
        """
        load_c_macros = list()
        alpha_beta_type = self.get_result_scale_type()
        nr_list = self.params.get_nr_list()
        for nr in nr_list:
            mr_list = self.params.get_mr_list(nr)
            for mr in mr_list:
                if len(c_row_addr_temp_regs) < mr:
                    raise NotImplementedError(
                        f"当前不支持mr={mr}大于临时寄存器数量的情况，请增加临时寄存器数量"
                    )
                load_instructions = []
                if alpha_beta_type.value in [
                    ResultScaleType.ALPHA1_BETA0.value,
                    ResultScaleType.ALPHA_BETA0.value,
                ]:
                    load_instructions.extend(
                        self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs)
                    )
                    load_instructions.extend(self.__generate_load_c_with_clear(mr, nr))
                elif alpha_beta_type.value in [
                    ResultScaleType.ALPHA1_BETA1.value,
                    ResultScaleType.ALPHA_BETA1.value,
                ]:
                    load_instructions.extend(
                        self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs)
                    )
                    load_instructions.extend(
                        self.__generate_load_c_with_beta_scale(
                            mr, nr, c_row_addr_temp_regs, only_support_beta_one=True
                        )
                    )
                else:
                    load_instructions.extend(
                        self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs)
                    )
                    load_instructions.extend(
                        self.__generate_load_c_with_beta_scale(
                            mr, nr, c_row_addr_temp_regs, only_support_beta_one=False
                        )
                    )

                load_c_macros.append(
                    {
                        "name": load_c_macro_fmt.substitute(nr=nr, mr=mr),
                        "instructions": load_instructions,
                    }
                )
        return load_c_macros

    def generate_loop_k(
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
                kr_list = self.params.get_kr_list(mr=mr, nr=nr)
                kr_list.append(0)  # 方便生成最后一个循环
                for kr, next_kr in pairwise(kr_list):
                    # loop_k_instructions.append(f"loop_nr{nr}_mr{mr}_loop_kr{kr}_update:    // 更新剩余K维度计数器")
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

    def generate_loop_m(
        self,
        mr_temp_reg: AllocatedGeneralRegister,
        loop_m_macro_fmt: Template,
        loop_k_macro_fmt: Template,
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

    def generate_loop_n(
        self, n_temp_reg: AllocatedGeneralRegister, loop_m_macro_fmt: Template
    ) -> List[str]:
        """
        生成N维度循环
        """
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        nr_list = self.params.get_nr_list()
        main_loop_n_instructions = [
            current_indent + f"// N维度主循环体: 处理nr in {json.dumps(nr_list)}"
        ]
        nr_list.append(0)  # 方便生成最后一个循环
        for nr, next_nr in pairwise(nr_list):
            # 计算当前剩余未处理的N维度大小
            main_loop_n_instructions.append(
                f"MAIN_LOOP_NR{nr}_LOOP_UPDATE_{label_suffix}:    // 更新剩余N维度计数器"
            )
            main_loop_n_instructions.append(
                current_indent
                + f"sub {n_temp_reg.reg_name}, {self.register_allocation.dim_reg_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}   // 计算剩余未处理的N维度"
            )
            main_loop_n_instructions.append(
                f"MAIN_LOOP_NR{nr}_START_{label_suffix}:         // 处理nr={nr}的主循环开始"
            )
            main_loop_n_instructions.append(
                current_indent
                + f"cmp {n_temp_reg.reg_name}, #{nr}       // 比较未处理的N维度与当前nr={nr}"
            )
            if next_nr != 0:
                main_loop_n_instructions.append(
                    current_indent
                    + f"blt MAIN_LOOP_NR{next_nr}_START_{label_suffix}   // 如果未处理N维度小于当前nr={nr}，跳转到下一个nr={next_nr}循环"
                )
            else:
                main_loop_n_instructions.append(
                    current_indent
                    + f"blt MAIN_LOOP_END_{label_suffix}   // 如果未处理N维度小于nr={nr}，跳转到主循环结束"
                )
            # 计算矩阵B和矩阵C的地址
            main_loop_n_instructions.append(
                current_indent
                + f"add {self.register_allocation.pointer_reg_b_for_current_block_base.reg_name}, {self.register_allocation.pointer_reg_b.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, LSL #{2}   // 计算B块N维度的基地址"
            )
            main_loop_n_instructions.append(
                current_indent
                + f"add {self.register_allocation.pointer_reg_c_for_loop_n.reg_name}, {self.register_allocation.pointer_reg_c.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, LSL #{2}   // 计算C块N维度的基地址"
            )
            # 调用M循环体
            main_loop_n_instructions.append(
                current_indent
                + f"{loop_m_macro_fmt.substitute(nr=nr)}            // 调用M维度循环体，处理nr={nr}"
            )
            # 更新已处理的N维度计数器
            main_loop_n_instructions.append(
                current_indent
                + f"add {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, #{nr}   // 更新已处理的N维度计数器"
            )
            main_loop_n_instructions.append(
                current_indent
                + f"b MAIN_LOOP_NR{nr}_LOOP_UPDATE_{label_suffix}   // 回到当前nr={nr}循环开始处"
            )
        return main_loop_n_instructions

    def _generate_a_row_addr(
        self, mr: int, row_addr: List[AllocatedGeneralRegister], with_prfm: bool = False
    ) -> List[str]:
        """
        生成计算矩阵A行地址的指令
        """
        instructions = []
        current_indent = "    "
        instructions.append(current_indent + f"// 计算矩阵A每行的地址: mr={mr}")
        for r in range(mr):
            instructions.append(current_indent + f"// 计算矩阵A第{r}行地址并装载")
            a_row_addr_temp_reg = row_addr[r]
            if r == 0:
                # instructions.append(current_indent + f"mov {a_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_a_for_current_block_base.reg_name}   // 计算A矩阵第{r}行首地址")
                instructions.append(
                    current_indent
                    + f"add {a_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_a_for_current_block_base.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, LSL #{self.params.get_dtype_lsl_position()}   // 调整A矩阵第{r}行地址到当前K维度位置"
                )
            else:
                last_a_row_addr_temp_reg = row_addr[r - 1]
                instructions.append(
                    current_indent
                    + f"add {a_row_addr_temp_reg.reg_name}, {last_a_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_lda.reg_name}   // 计算A矩阵第{r}行地址"
                )
            if with_prfm and (self.params.a_prefetch != PrefetchCacheLevel.NONE.value):
                instructions.append(
                    current_indent
                    + f"prfm PLD{self.params.a_prefetch.value.upper()}STRM, [{a_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_a.reg_name}]   // 预取A矩阵第{r}行的数据"
                )
        return instructions

    def _update_dealt_k_for_dim_k(self, kr: int) -> List[str]:
        """
        更新已处理的K维度计数器
        """
        current_indent = "    "
        instructions = [
            current_indent + f"// 更新已处理的K维度计数器: kr={kr}",
            current_indent
            + f"add {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, #{kr}   // 更新已处理的K维度计数器",
        ]
        return instructions

    def _generate_b_row_addr(
        self,
        kr: int,
        row_addr: List[AllocatedGeneralRegister],
        with_prfm: bool = False,
        row_idx: Optional[int] = None,
    ) -> List[str]:
        """
        生成计算矩阵B行地址的指令
        """
        instructions = []
        current_indent = "    "
        if row_idx is not None and row_idx >= kr:
            raise ValueError(f"row_idx={row_idx} 超出当前kr={kr}范围")

        if row_idx is None:
            instructions.append(current_indent + f"// 计算矩阵B每行的地址: kr={kr}")
            row_addr_num = len(row_addr)
            for k in range(row_addr_num):
                instructions.append(current_indent + f"// 计算矩阵B第{k}行地址并装载")
                b_row_addr_temp_reg = row_addr[k]
                if k == 0:
                    # instructions.append(current_indent + f"mov {b_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_b_for_current_block_base.reg_name}   // 计算B矩阵第{k}行首地址")
                    instructions.append(
                        current_indent
                        + f"madd {b_row_addr_temp_reg.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.stride_reg_ldb.reg_name}, {self.register_allocation.pointer_reg_b_for_current_block_base.reg_name}  // 调整B矩阵第{k}行地址到当前K维度位置"
                    )
                else:
                    last_b_row_addr_temp_reg = row_addr[k - 1]
                    instructions.append(
                        current_indent
                        + f"add {b_row_addr_temp_reg.reg_name}, {last_b_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldb.reg_name}   // 计算B矩阵第{k}行地址"
                    )

                if with_prfm and (
                    self.params.b_prefetch != PrefetchCacheLevel.NONE.value
                ):
                    instructions.append(
                        current_indent
                        + f"prfm PLD{self.params.b_prefetch.value.upper()}STRM, [{b_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_b.reg_name}]   // 预取B矩阵第{k}行的数据"
                    )
            return instructions
        else:
            instructions.append(
                current_indent + f"// 计算矩阵B第{row_idx}行的地址: kr={kr}"
            )
            row_addr_num = len(row_addr)
            b_row_addr_temp_reg = row_addr[row_idx % row_addr_num]  # 选择地址寄存器
            instructions.append(
                current_indent
                + f"add {b_row_addr_temp_reg.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, #{row_idx}         // 统计偏移的行数"
            )
            instructions.append(
                current_indent
                + f"madd {b_row_addr_temp_reg.reg_name}, {b_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldb.reg_name}, {self.register_allocation.pointer_reg_b_for_current_block_base.reg_name}  // 调整B矩阵第{row_idx}行地址到当前K维度位置"
            )
            if with_prfm and (self.params.b_prefetch != PrefetchCacheLevel.NONE.value):
                instructions.append(
                    current_indent
                    + f"prfm PLD{self.params.b_prefetch.value.upper()}STRM, [{b_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_b.reg_name}]   // 预取B矩阵第{row_idx}行的数据"
                )
            return instructions

    def _get_vector_reg_scheduler_for_kernel_compute(
        self, mr: int, kr: int, nr: int
    ) -> Tuple[
        List[List[AllocatedVectorRegister]], List[List[AllocatedVectorRegister]]
    ]:
        """
        获取内核计算体的向量寄存器调度器
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        mat_a_vec_regs_scheduler = []
        mat_b_vec_regs_scheduler = []
        # 先分配矩阵B使用的向量寄存器
        mat_b_rows, mat_b_cols = kr, ceil(nr / elements_per_vector_reg)
        self.register_allocation.b_vector_regs.clear()
        self.register_allocation.b_vector_regs.set_infinite_iteration(True)
        mat_b_vec_allocator = self.register_allocation.b_vector_regs
        for r in range(mat_b_rows):
            col_scheduler = []
            for c in range(mat_b_cols):
                col_scheduler.append(next(mat_b_vec_allocator))
            mat_b_vec_regs_scheduler.append(col_scheduler)
        # 再分配矩阵A使用的向量寄存器
        mat_a_rows, mat_a_cols = mr, ceil(kr / elements_per_vector_reg)
        self.register_allocation.a_vector_regs.clear()
        self.register_allocation.a_vector_regs.set_infinite_iteration(True)
        mat_a_vec_allocator = self.register_allocation.a_vector_regs
        for r in range(mat_a_cols):
            row_scheduler = []
            for c in range(mat_a_rows):
                row_scheduler.append(next(mat_a_vec_allocator))
            mat_a_vec_regs_scheduler.append(row_scheduler)
        return mat_a_vec_regs_scheduler, mat_b_vec_regs_scheduler

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
        #     if mr > 1:
        #         instructions.append(
        #             (
        #                 [],
        #                 current_indent + f"// 计算开结束: mr={mr}, kr={kr}, nr={nr}, round_idx={round_idx}",
        #                 deepcopy(mat_b_round_scheduler),
        #                 []
        #             )
        #         )
        return instructions

    def generate_kernel_compute(
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
                f"/**** 内核说明",
                current_indent + f"* 内核大小: mr={mr}, kr={kr}, nr={nr}",
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
                + f"blt KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_FINISH_{label_suffix}    // 如果剩余K维度小于2*kr={2 * kr}，跳转到最后一轮计算"
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
                f"KERNEL_COMPUTE_NR{nr}_MR{mr}_KR{kr}_PINGPONG_{label_suffix} :    // 内核计算体Ping-Pong计算开始, mr={mr}, kr={kr}, nr={nr}"
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
        loop_k_macro_fmt = Template("LOOP_NR${nr}_MR${mr}_KR" + f"_{label_suffix}")
        main_loop_m_instructions = self.generate_loop_m(
            m_temp_reg, loop_m_macro_fmt, loop_k_macro_fmt
        )  # 生成M循环体

        # 处理K维度的循环
        store_c_macro_fmt = Template("STORE_C_NR${nr}_MR${mr}" + f"_{label_suffix}")
        load_c_macro_fmt = Template("LOAD_C_NR${nr}_MR${mr}" + f"_{label_suffix}")
        main_loop_k_instructions = self.generate_loop_k(
            k_temp_reg,
            loop_k_macro_fmt,
            self.get_kernel_macro_fmt(),
            load_c_macro_fmt,
            store_c_macro_fmt,
        )  # 生成K循环体

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
        kernel_compute_instructions = self.generate_kernel_compute(
            self.get_kernel_macro_fmt(), available_temp_regs
        )

        return {
            "main_loop_n": main_loop_n_instructions,
            "main_loop_m": main_loop_m_instructions,
            "main_loop_k": main_loop_k_instructions,
            "load_c": load_c_instructions,
            "store_c": store_c_instructions,
            "kernel_compute": kernel_compute_instructions,
        }

    def generate_main_loop_scheduler(self) -> List[str]:
        """
        生成主计算循环调度器
        """
        instructions = list()
        current_indent = "    "
        temp_vec_reg = AllocatedVectorRegister(
            var_name="temp", reg_name="v31", precision=self.params.data_type
        )  # 临时使用, 入口判断完就不再使用
        instructions.append(
            current_indent
            + f"fmov {temp_vec_reg.s_reg_name}, #1.0               // 准备常数1.0, 该寄存器在入口判断完就不再使用"
        )
        instructions.append(
            current_indent
            + f"ins {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[0], {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]  // 根据alpha的值判断是否需要缩放AB"
        )
        instructions.append(
            current_indent
            + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较alpha与1.0"
        )
        instructions.append(
            current_indent
            + f"beq main_loop_with_alpha_one             // 如果alpha为1.0，继续后面的判断"
        )

        instructions.append(
            current_indent
            + f"ins {self.register_allocation.scalar_reg_beta.v_reg_name}.s[0], {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]  // 根据beta值判断是否需要装载C矩阵块"
        )
        instructions.append(
            current_indent
            + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, #0.0                // 比较beta与0.0"
        )
        instructions.append(
            current_indent
            + f"beq main_loop_with_alpha_beta_zero      // alpha=*, beta=0.0"
        )
        instructions.append(
            current_indent
            + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较beta与1.0"
        )
        instructions.append(
            current_indent
            + f"beq main_loop_with_alpha_beta_one      // alpha=*, beta=1.0"
        )

        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha_beta_name()}      // alpha=*, beta=*"
        )
        instructions.append(
            current_indent + "b main_loop_scheduler_end      // 跳到结束"
        )

        instructions.append(
            "main_loop_with_alpha_beta_zero:            // alpha=*, beta=0.0"
        )
        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha_beta0_name()}      // alpha=*, beta=0.0"
        )
        instructions.append(
            current_indent + "b main_loop_scheduler_end      // 跳到结束"
        )

        instructions.append(
            "main_loop_with_alpha_beta_one:             // alpha=*, beta=1.0"
        )
        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha_beta1_name()}      // alpha=*, beta=1.0"
        )
        instructions.append(
            current_indent + "b main_loop_scheduler_end      // 跳到结束"
        )

        instructions.append(
            f"main_loop_with_alpha_one:             // 如果alpha为1.0，继续后面的判断"
        )
        instructions.append(
            current_indent
            + f"mov {self.register_allocation.scalar_reg_beta.v_reg_name}.s[0], {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]  // 根据beta值判断是否需要装载C矩阵块"
        )
        instructions.append(
            current_indent
            + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, #0.0                // 比较beta与0.0"
        )
        instructions.append(
            current_indent
            + f"beq main_loop_with_alpha_one_beta_zero      // alpha=1, beta=0.0"
        )
        instructions.append(
            current_indent
            + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较beta与1.0"
        )
        instructions.append(
            current_indent
            + f"beq main_loop_with_alpha_one_beta_one      // alpha=1, beta=1.0"
        )

        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha1_beta_name()}      // alpha=1, beta=*"
        )
        instructions.append(
            current_indent + "b main_loop_scheduler_end      // 跳到结束"
        )

        instructions.append(
            "main_loop_with_alpha_one_beta_zero:            // alpha=1, beta=0.0"
        )
        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha1_beta0_name()}      // alpha=1, beta=0.0"
        )
        instructions.append(
            current_indent + "b main_loop_scheduler_end      // 跳到结束"
        )

        instructions.append(
            "main_loop_with_alpha_one_beta_one:             // alpha=1, beta=1.0"
        )
        instructions.append(
            current_indent
            + f"{self.get_main_loop_with_alpha1_beta1_name()}      // alpha=1, beta=1.0"
        )

        instructions.append(
            current_indent + f"main_loop_scheduler_end:             // 调度结束"
        )

        return instructions

    def get_main_loop_with_alpha1_beta0(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=0"""
        return self.generate_main_loop_logic()

    def get_main_loop_with_alpha1_beta1(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=1"""
        return self.generate_main_loop_logic()

    def get_main_loop_with_alpha1_beta(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=*"""
        return self.generate_main_loop_logic()

    def get_main_loop_with_alpha_beta0(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=0"""
        return self.generate_main_loop_logic()

    def get_main_loop_with_alpha_beta1(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=1"""
        return self.generate_main_loop_logic()

    def get_main_loop_with_alpha_beta(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=*"""
        return self.generate_main_loop_logic()
