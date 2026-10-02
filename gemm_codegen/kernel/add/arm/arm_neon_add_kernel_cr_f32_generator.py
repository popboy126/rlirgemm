# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/29 09:12
# Contact       : hucheng.lew@qq.com
# File          : arm_neon_add_kernel_cr_ar_br_f32_generator.py
# IDE           : PyCharm
# Description   : ARM NEON加法内核代码生成器（CR-AR-BR布局，F32精度）

from .base_arm_neon_add_kernel_generator import BaseArmNeonAddKernelGenerator
from typing import List, Tuple, Dict, Any, Optional
from gemm_codegen.kernel.add.base import (
    KernelParameters, RegisterAllocation,
    AllocatedVectorRegister, AllocatedVectorRegisters,
    AllocatedGeneralRegister, AllocatedGeneralRegisters,
)

from gemm_codegen.common import (
    MatrixLayout,
    PrefetchCacheLevel,
    ResultScaleType,
    PrecisionType,
    flatten_list
)
from math import ceil, floor
from itertools import pairwise, chain, zip_longest
from string import Template
import json



class ArmNeonAddKernelCrF32Generator(BaseArmNeonAddKernelGenerator):
    """
    采用ARM NEON指令实现的矩阵加法内核生成器（C行主、部分和行主）
    """
    def __init__(self, params: KernelParameters):
        super().__init__(params)

        if self.params.data_type.precision.value != PrecisionType.FP32.value:
            raise ValueError("ArmNeonAddKernelCrF32Generator 仅支持 f32 数据类型")

        if not (self.params.layout.value == MatrixLayout.ROW_MAJOR.value):
            raise ValueError("ArmNeonAddKernelCrF32Generator 仅支持 C、部分和 均为行主序布局")

    def allocate_registers(self) -> RegisterAllocation:
        """
        分配寄存器
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        # 分配C的向量寄存器
        c_vec_regs_needed = self.params.mr * ceil(self.params.nr / elements_per_vector_reg)

        partial_sum_row_reg_count = ceil(self.params.nr / elements_per_vector_reg)
        remained_reg_nums = self.get_vector_register_count() - c_vec_regs_needed - 1  # 减去alpha和beta占用的寄存器
        if self.params.kr * self.params.mr * partial_sum_row_reg_count <= remained_reg_nums:
            partial_sum_row_regs_needed = self.params.kr * self.params.mr * partial_sum_row_reg_count
        elif self.params.kr * self.params.mr * self.params.vec_reg_reuse_interval <= remained_reg_nums:
            partial_sum_row_regs_needed = self.params.kr * self.params.mr * self.params.vec_reg_reuse_interval
        else:
            raise ValueError(f"寄存器分配失败: 剩余向量寄存器数量不足以分配部分和寄存器, 需要至少 {self.params.kr * self.params.mr * self.params.vec_reg_reuse_interval} 个寄存器, 但只剩余 {remained_reg_nums} 个寄存器")

        # 向量寄存器分配生成器
        vector_register_allocator = self.vector_register_allocator()
        general_register_allocator = self.general_register_allocator()
        return RegisterAllocation(
            c_vector_regs = AllocatedVectorRegisters(
                var_name='C',
                reg_names = list(reversed([next(vector_register_allocator) for _ in range(c_vec_regs_needed)])),
                precision=self.params.data_type
            ),
            partial_sum_vector_regs = AllocatedVectorRegisters(
                var_name='partial_sum',
                reg_names = list(reversed([next(vector_register_allocator) for _ in range(partial_sum_row_regs_needed)])),
                precision=self.params.data_type
            ),

            # 注意通用寄存器的分配要跟函数的调用规则匹配上
            pointer_reg_c=AllocatedGeneralRegister(
                var_name="C",
                reg_name=next(general_register_allocator)
            ),
            stride_reg_ldc = AllocatedGeneralRegister(
                var_name="ldc",
                reg_name=next(general_register_allocator)
            ),

            pointer_reg_partial_sum=AllocatedGeneralRegister(
                var_name="partial_sum",
                reg_name=next(general_register_allocator)
            ),
            stride_reg_ldp = AllocatedGeneralRegister(
                var_name="ldp",
                reg_name=next(general_register_allocator),
            ),

            dim_reg_m=AllocatedGeneralRegister(
                var_name="M",
                reg_name=next(general_register_allocator)
            ),
            dim_reg_n=AllocatedGeneralRegister(
                var_name="N",
                reg_name=next(general_register_allocator)
            ),
            dim_reg_k=AllocatedGeneralRegister(
                var_name="K",
                reg_name=next(general_register_allocator)
            ),

            scalar_reg_alpha = AllocatedVectorRegister(
                var_name="alpha",
                reg_name='v0',
                precision=self.params.data_type,
                position=2
            ),
            scalar_reg_beta = AllocatedVectorRegister(
                var_name="beta",
                reg_name='v0',
                precision=self.params.data_type,
                position=3
            ),

            prefetch_reg_c = AllocatedGeneralRegister(
                var_name="prefetch_c_stride",
                reg_name=next(general_register_allocator)
            ),
            prefetch_reg_partial_sum = AllocatedGeneralRegister(
                var_name="prefetch_partial_sum_stride",
                reg_name=next(general_register_allocator)
            ),

            loop_reg_dealt_m = AllocatedGeneralRegister(
                var_name="dealt_m",
                reg_name=next(general_register_allocator)
            ),
            loop_reg_dealt_n = AllocatedGeneralRegister(
                var_name="dealt_n",
                reg_name=next(general_register_allocator)
            ),
            loop_reg_dealt_k = AllocatedGeneralRegister(
                var_name="dealt_k",
                reg_name=next(general_register_allocator)
            ),
            pointer_reg_c_row_base= AllocatedGeneralRegister(
                var_name="c_block_row_base",
                reg_name=next(general_register_allocator)
            ),
            row_offset_partial_sum_base= AllocatedGeneralRegister(
                var_name="partial_sum_block_row_offset",
                reg_name=next(general_register_allocator)
            ),
            pointer_reg_c_for_current_block_base = AllocatedGeneralRegister(
                var_name="c_block_base",
                reg_name=next(general_register_allocator)
            ),
            offset_reg_partial_sum_for_current_block = AllocatedGeneralRegister(
                var_name="partial_sum_block_offset",
                reg_name=next(general_register_allocator)
            ),

            temp_regs = AllocatedGeneralRegisters(
                regs = [ AllocatedGeneralRegister(
                    var_name=f"temp_reg_{idx}",
                    reg_name=reg
                ) for idx, reg in enumerate(general_register_allocator)]  # 剩余的全部分配
            )

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

        kernel_shapes.add((mr, kr, nr))
        if self.params.support_edge_opt:
            # 计算nr的边界
            if nr > elements_per_vector_reg:
                n_list = [1, 2]
                n_list.extend([i for i in range(elements_per_vector_reg, nr + 1, elements_per_vector_reg)])
            elif nr == elements_per_vector_reg:
                n_list = [1, 2, elements_per_vector_reg]
            else:
                n_list = [1, 2]
            # 计算kr的边界
            if kr > elements_per_vector_reg:
                k_list = [1, 2]
                k_list.extend([i for i in range(elements_per_vector_reg, kr + 1, elements_per_vector_reg)])
            elif kr == elements_per_vector_reg:
                k_list = [1, 2, elements_per_vector_reg]
            else:
                k_list = [1, 2]

            for m in range(1, mr + 1):
                for n in n_list:
                    for k in k_list:
                        kernel_shapes.add((m, k, n))

        return sorted(list(kernel_shapes), key=lambda x: (x[0], x[2], x[1]), reverse=True)

    def generate_main_loop_scheduler(self) -> List[str]:
        """
        生成主计算循环调度器
        """
        instructions = list()
        current_indent = "    "
        temp_vec_reg = AllocatedVectorRegister(var_name="temp", reg_name="v31", precision=self.params.data_type)  # 临时使用, 入口判断完就不再使用
        instructions.append(current_indent + f"fmov {temp_vec_reg.s_reg_name}, #1.0               // 准备常数1.0, 该寄存器在入口判断完就不再使用")
        instructions.append(current_indent + f"ins {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[0], {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]  // 根据alpha的值判断是否需要缩放AB")
        instructions.append(current_indent + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较alpha与1.0")
        instructions.append(current_indent + f"beq main_loop_with_alpha_one             // 如果alpha为1.0，继续后面的判断")

        instructions.append(current_indent + f"ins {self.register_allocation.scalar_reg_beta.v_reg_name}.s[0], {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]  // 根据beta值判断是否需要装载C矩阵块")
        instructions.append(current_indent + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, #0.0                // 比较beta与0.0")
        instructions.append(current_indent + f"beq main_loop_with_alpha_beta_zero      // alpha=*, beta=0.0")
        instructions.append(current_indent + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较beta与1.0")
        instructions.append(current_indent + f"beq main_loop_with_alpha_beta_one      // alpha=*, beta=1.0")

        instructions.append(current_indent + f"{self.get_main_loop_with_alpha_beta_name()}      // alpha=*, beta=*")
        instructions.append(current_indent + "b main_loop_scheduler_end      // 跳到结束")

        instructions.append("main_loop_with_alpha_beta_zero:            // alpha=*, beta=0.0")
        instructions.append(current_indent + f"{self.get_main_loop_with_alpha_beta0_name()}      // alpha=*, beta=0.0")
        instructions.append(current_indent + "b main_loop_scheduler_end      // 跳到结束")

        instructions.append("main_loop_with_alpha_beta_one:             // alpha=*, beta=1.0")
        instructions.append(current_indent + f"{self.get_main_loop_with_alpha_beta1_name()}      // alpha=*, beta=1.0")
        instructions.append(current_indent + "b main_loop_scheduler_end      // 跳到结束")

        instructions.append(f"main_loop_with_alpha_one:             // 如果alpha为1.0，继续后面的判断")
        instructions.append(current_indent + f"mov {self.register_allocation.scalar_reg_beta.v_reg_name}.s[0], {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]  // 根据beta值判断是否需要装载C矩阵块")
        instructions.append(current_indent + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, #0.0                // 比较beta与0.0")
        instructions.append(current_indent + f"beq main_loop_with_alpha_one_beta_zero      // alpha=1, beta=0.0")
        instructions.append(current_indent + f"fcmp {self.register_allocation.scalar_reg_beta.s_reg_name}, {temp_vec_reg.s_reg_name}                // 比较beta与1.0")
        instructions.append(current_indent + f"beq main_loop_with_alpha_one_beta_one      // alpha=1, beta=1.0")

        instructions.append(current_indent + f"{self.get_main_loop_with_alpha1_beta_name()}      // alpha=1, beta=*")
        instructions.append(current_indent + "b main_loop_scheduler_end      // 跳到结束")

        instructions.append("main_loop_with_alpha_one_beta_zero:            // alpha=1, beta=0.0")
        instructions.append(current_indent + f"{self.get_main_loop_with_alpha1_beta0_name()}      // alpha=1, beta=0.0")
        instructions.append(current_indent + "b main_loop_scheduler_end      // 跳到结束")

        instructions.append("main_loop_with_alpha_one_beta_one:             // alpha=1, beta=1.0")
        instructions.append(current_indent + f"{self.get_main_loop_with_alpha1_beta1_name()}      // alpha=1, beta=1.0")

        instructions.append(current_indent + f"main_loop_scheduler_end:             // 调度结束")

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


    def generate_main_loop_logic(self) -> Dict[str, List[str] | Any]:
        """生成主计算循环"""
        self.register_allocation.clear()
        main_loop_m_instructions = []
        current_indent = "    "
        label_suffix = self.get_label_suffix()

        main_loop_m_instructions.extend([
            current_indent + f"eor {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}  // 清零M维度计数器",
            current_indent + f"lsl {self.register_allocation.stride_reg_ldc.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}, #{self.params.get_dtype_lsl_position()}    // ldc*{1<<self.params.get_dtype_lsl_position()}转换为字节偏移",
            current_indent + f"lsl {self.register_allocation.stride_reg_ldp.reg_name}, {self.register_allocation.stride_reg_ldp.reg_name}, #{self.params.get_dtype_lsl_position()}    // ldp*{1<<self.params.get_dtype_lsl_position()}转换为字节偏移",
            ])
        mr_list = self.params.get_mr_list()
        mr_list.append(0)  # 方便生成最后一个循环
        self.register_allocation.temp_regs.set_infinite_iteration(True)
        m_temp_reg = next(self.register_allocation.temp_regs)
        n_temp_reg = next(self.register_allocation.temp_regs)
        k_temp_reg = next(self.register_allocation.temp_regs)
        loop_n_macro_fmt = Template("LOOP_MR${mr}_NR"+ f"_{label_suffix}")
        main_loop_m_instructions.extend(self.generate_loop_m(m_temp_reg, loop_n_macro_fmt))   # 生成M循环体
        main_loop_m_instructions.append(f"MAIN_LOOP_END_{label_suffix}:    // M维度(主)循环结束标记")

        # 处理M维度的循环
        loop_k_macro_fmt = Template("LOOP_MR${mr}_NR${nr}_KR" + f"_{label_suffix}")
        main_loop_n_instructions = self.generate_loop_n(n_temp_reg, loop_n_macro_fmt, loop_k_macro_fmt)  # 生成M循环体
        #
        # 处理K维度的循环
        store_c_macro_fmt = Template("STORE_C_MR${mr}_NR${nr}" + f"_{label_suffix}")
        load_c_macro_fmt = Template("LOAD_C_MR${mr}_NR${nr}"+ f"_{label_suffix}")
        main_loop_k_instructions = self.generate_loop_k(k_temp_reg, loop_k_macro_fmt, self.get_kernel_macro_fmt(), load_c_macro_fmt, store_c_macro_fmt)  # 生成K循环体

        c_row_addr_temp_regs = [ next(self.register_allocation.temp_regs) for _ in range(self.params.mr)]
        load_c_instructions = self.generate_load_c(load_c_macro_fmt, c_row_addr_temp_regs)
        store_c_instructions = self.generate_store_c(store_c_macro_fmt, c_row_addr_temp_regs)

        addition_reg_num = self.params.mr + self.params.kr + 1 - len(c_row_addr_temp_regs) + 2  # 行地址计算, kr个部分和地址计算, 1个用于部分和累加，1个用于数据预取地址的计算, 1个用于计算部分和指针偏移量的寄存器
        available_temp_regs = c_row_addr_temp_regs + [
            next(self.register_allocation.temp_regs) for _ in range(addition_reg_num)  # 用于地址计算: 各行偏移量, 每个部分和的初始地址
        ]
        kernel_compute_instructions = self.generate_kernel_compute(self.get_kernel_macro_fmt(), available_temp_regs)

        return {
            "main_loop_m": main_loop_m_instructions,
            "main_loop_n": main_loop_n_instructions,
            "main_loop_k": main_loop_k_instructions,
            "load_c": load_c_instructions,
            "store_c": store_c_instructions,
            "kernel_compute": kernel_compute_instructions
        }

    def generate_loop_m(self, m_temp_reg: AllocatedGeneralRegister, loop_n_macro_fmt:Template) -> List[str]:
        """
        生成M维度循环
        """
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        mr_list = self.params.get_mr_list()
        main_loop_n_instructions = [
            current_indent + f"// M维度主循环体: 处理nr in {json.dumps(mr_list)}"
        ]
        mr_list.append(0)  # 方便生成最后一个循环
        for mr, next_mr in pairwise(mr_list):
            # 计算当前剩余未处理的M维度大小
            main_loop_n_instructions.append(f"MAIN_LOOP_MR{mr}_LOOP_UPDATE_{label_suffix}:    // 更新剩余M维度计数器")
            main_loop_n_instructions.append(current_indent + f"sub {m_temp_reg.reg_name}, {self.register_allocation.dim_reg_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}   // 计算剩余未处理的M维度")
            main_loop_n_instructions.append(f"MAIN_LOOP_MR{mr}_START_{label_suffix}:         // 处理mr={mr}的主循环开始")
            main_loop_n_instructions.append(current_indent + f"cmp {m_temp_reg.reg_name}, #{mr}       // 比较未处理的M维度与当前mr={mr}")
            if next_mr != 0:
                main_loop_n_instructions.append(current_indent + f"blt MAIN_LOOP_MR{next_mr}_START_{label_suffix}   // 如果未处理M维度小于当前mr={mr}，跳转到下一个nr={next_mr}循环")
            else:
                main_loop_n_instructions.append(current_indent + f"blt MAIN_LOOP_END_{label_suffix}   // 如果未处理M维度小于mr={mr}，跳转到主循环结束")
            # 计算矩阵C的块的基地址(行)
            main_loop_n_instructions.append(current_indent + f"madd {self.register_allocation.pointer_reg_c_row_base.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.pointer_reg_c.reg_name}   // 计算C块基地址的行地址")
            # 计算部分和的偏移地址(行)
            main_loop_n_instructions.append(current_indent + f"mul {self.register_allocation.row_offset_partial_sum_base.reg_name}, {self.register_allocation.stride_reg_ldp.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}  // 计算部分和块的偏移地址(行)")
            # 调用N循环体
            main_loop_n_instructions.append(current_indent + f"{loop_n_macro_fmt.substitute(mr=mr)}            // 调用N维度循环体，处理mr={mr}")
            # 更新已处理的M维度计数器
            main_loop_n_instructions.append(current_indent + f"add {self.register_allocation.loop_reg_dealt_m.reg_name}, {self.register_allocation.loop_reg_dealt_m.reg_name}, #{mr}   // 更新已处理的M维度计数器")
            main_loop_n_instructions.append(current_indent + f"b MAIN_LOOP_MR{mr}_LOOP_UPDATE_{label_suffix}   // 回到当前mr={mr}循环开始处")
        return main_loop_n_instructions

    def generate_loop_n(self, nr_temp_reg: AllocatedGeneralRegister, loop_n_macro_fmt:Template, loop_k_macro_fmt:Template) -> List[Dict[str, List[str] | Any]]:
        """生成N维度循环"""
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        mr_list = self.params.get_mr_list()
        loop_n_macros = list()
        for mr in mr_list:
            loop_n_instructions = []
            loop_n_instructions.append(current_indent + f"// N维度循环体: 处理mr={mr}, nr={self.params.nr}, kr={self.params.kr}")
            loop_n_instructions.append(current_indent + f"// 实现mr={mr}, nr={self.params.nr} 的M循环计算逻辑")
            loop_n_instructions.append(current_indent + f"eor {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}  // 清零N维度计数器")
            nr_list = self.params.get_nr_list(mr)
            nr_list.append(0)  # 方便生成最后一个循环
            for nr, next_nr in pairwise(nr_list):
                # 计算当前剩余未处理的M维度大小
                loop_n_instructions.append(f"LOOP_MR{mr}_LOOP_NR{nr}_UPDATE_{label_suffix}:    // 更新剩余N维度计数器")
                loop_n_instructions.append(current_indent + f"sub {nr_temp_reg.reg_name}, {self.register_allocation.dim_reg_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}   // 计算剩余未处理的N维度")
                loop_n_instructions.append(f"LOOP_MR{mr}_LOOP_NR{nr}_START_{label_suffix}:         // 处理mr={mr}的N循环开始, 此时nr={nr}")
                loop_n_instructions.append(current_indent + f"cmp {nr_temp_reg.reg_name}, #{nr}       // 比较未处理N维度与当前nr={nr}")
                if next_nr != 0:
                    loop_n_instructions.append(current_indent + f"blt LOOP_MR{mr}_LOOP_NR{next_nr}_START_{label_suffix}   // 如果未处理N维度小于当前nr={nr}，跳转到下一个nr={next_nr}循环, 此时mr={mr}")
                else:
                    loop_n_instructions.append(current_indent + f"blt LOOP_MR{mr}_LOOP_NR_END_{label_suffix}   // 如果未处理N维度小于nr={nr}，mr下又没有更小的nr，那么跳转到N循环结束, 此时mr={mr}")
                # 计算矩阵C的块的基地址
                loop_n_instructions.append(current_indent + f"add {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}, {self.register_allocation.pointer_reg_c_row_base.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, LSL #{self.params.get_dtype_lsl_position()}   // 计算C块的基地址")
                # 计算部分和的偏移地址
                loop_n_instructions.append(current_indent + f"add {self.register_allocation.offset_reg_partial_sum_for_current_block.reg_name}, {self.register_allocation.row_offset_partial_sum_base.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, LSL #{self.params.get_dtype_lsl_position()}   // 计算部分和的偏移地址量")

                # 调用M循环体
                loop_n_instructions.append(current_indent + f"{loop_k_macro_fmt.substitute(mr=mr, nr=nr)}            // 调用K维度循环体，处理mr={mr}, nr={nr}")
                # 记录处理过程
                # 更新已处理的N维度计数器
                loop_n_instructions.append(current_indent + f"add {self.register_allocation.loop_reg_dealt_n.reg_name}, {self.register_allocation.loop_reg_dealt_n.reg_name}, #{nr}   // 更新已处理的N维度计数器")
                loop_n_instructions.append(current_indent + f"b LOOP_MR{mr}_LOOP_NR{nr}_UPDATE_{label_suffix}    // 回到当前nr={nr}循环开始处, 此时mr={mr}")

            loop_n_instructions.append(f"LOOP_MR{mr}_LOOP_NR_END_{label_suffix} :           // N循环结束, 此时mr={mr}")
            loop_n_macros.append({
                "name": loop_n_macro_fmt.substitute(mr=mr),
                "instructions": loop_n_instructions,
            })
        return loop_n_macros


    def generate_loop_k(self, k_temp_reg: AllocatedGeneralRegister, loop_k_macro_fmt:Template, kernel_macro_fmt: Template, load_c_macro_fmt: Template, store_c_macro_fmt: Template) -> List[Dict[str, List[str] | Any]]:
        """生成K维度循环"""
        current_indent = "    "
        label_suffix = self.get_label_suffix()
        loop_k_macros = list()
        mr_list = self.params.get_mr_list()
        for mr in mr_list:
            nr_list = self.params.get_nr_list(mr)
            for nr in nr_list:
                loop_k_instructions = []
                loop_k_instructions.append(current_indent + f"// K维度循环体: 处理mr={mr}, nr={nr}, kr={self.params.kr}")
                loop_k_instructions.append(current_indent + f"// 实现mr={mr}, nr={nr} 的K循环计算逻辑")
                loop_k_instructions.append(current_indent + f"{load_c_macro_fmt.substitute(mr=mr, nr=nr)}        // 加载C矩阵块")
                loop_k_instructions.append(current_indent + f"eor {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}  // 清零K维度计数器")
                loop_k_instructions.append(current_indent + f"mov {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}   // 初始化剩余K维度计数器")
                kr_list = self.params.get_kr_list(mr=mr, nr=nr)
                kr_list.append(0)  # 方便生成最后一个循环
                for kr, next_kr in pairwise(kr_list):
                    loop_k_instructions.append(f"LOOP_MR{mr}_NR{nr}_LOOP_KR{kr}_START_{label_suffix} :         // 处理kr={kr}的K循环开始, 此时mr={mr}, nr={nr}")
                    loop_k_instructions.append(current_indent + f"cmp {k_temp_reg.reg_name}, #{kr}       // 比较未处理K维度与当前kr={kr}")
                    if next_kr != 0:
                        loop_k_instructions.append(current_indent + f"blt LOOP_MR{mr}_NR{nr}_LOOP_KR{next_kr}_START_{label_suffix}    // 如果未处理K维度小于当前kr={kr}，跳转到下一个kr={next_kr}循环, 此时mr={mr}, nr={nr}")
                    else:
                        loop_k_instructions.append(current_indent + f"blt LOOP_MR{mr}_NR{nr}_LOOP_KR_END_{label_suffix}    // 如果未处理K维度小于kr={kr}，nr下又没有更小的kr，那么跳转到K循环结束, 此时mr={mr}, nr={nr}")
                    # 调用K循环体
                    loop_k_instructions.append(current_indent + f"{kernel_macro_fmt.substitute(nr=nr, mr = mr, kr=kr)}            // 调用内核计算体，处理mr={mr}, nr={nr}, kr={kr}")
                    # 记录处理过程
                    # 计算当前剩余未处理的K维度大小
                    loop_k_instructions.append(current_indent + f"sub {k_temp_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 计算剩余未处理的K维度")
                loop_k_instructions.append(f"LOOP_MR{mr}_NR{nr}_LOOP_KR_END_{label_suffix} :           // K循环结束, 此时mr={mr}, nr={nr}")
                loop_k_instructions.append(current_indent + f"{store_c_macro_fmt.substitute(nr=nr, mr=mr)}        // 保存C矩阵块")
                loop_k_macros.append({
                    "name": loop_k_macro_fmt.substitute(nr=nr, mr=mr),
                    "instructions": loop_k_instructions,
                })
        return loop_k_macros

    def __generate_c_row_addr(self, mr: int, nr: int, c_row_addr_temp_regs: List[AllocatedGeneralRegister]) -> List[str]:
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
                instructions.append(current_indent + f"mov {c_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}   // 计算C矩阵第{r}行地址")
            else:
                last_c_row_addr_temp_reg = c_row_addr_temp_regs[r - 1]
                instructions.append(current_indent + f"add {c_row_addr_temp_reg.reg_name}, {last_c_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}   // 计算C矩阵第{r}行地址")
        return instructions

    def __generate_load_c_with_clear(self, mr: int, nr: int) -> List[str]:
        """
        生成装载C矩阵块并清零的指令
        """
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        rows, cols = mr, ceil(nr / vector_reg_element_width)
        used_vec_regs, idle_vec_regs = self.register_allocation.c_vector_regs.get_rearranged_reg_names(rows, cols)
        load_instructions = [
            current_indent + f"// 开始清空矩阵C的向量寄存器: mr={mr}, nr={nr}"
        ]
        for r in range(rows):
            for c in range(cols):
                load_instructions.append(current_indent + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 清零C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}列对应的向量寄存器")
        return load_instructions

    def __generate_load_c_with_beta_scale(self, mr: int, nr: int, c_row_addr_temp_regs: List[AllocatedGeneralRegister], only_support_beta_one: bool) -> List[str]:
        """
        生成装载C矩阵块并乘以beta的指令
        """
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        rows, cols = mr, ceil(nr / vector_reg_element_width)
        used_vec_regs, idle_vec_regs = self.register_allocation.c_vector_regs.get_rearranged_reg_names(rows, cols)
        load_instructions = [
            current_indent + f"// 开始装载矩阵C块的数据到向量寄存器: mr={mr}, nr={nr}"
        ]
        for r in range(rows):
            c_row_addr_temp_reg = c_row_addr_temp_regs[r]
            remained_nr = nr
            while remained_nr > 0:
                stored_nr = nr - remained_nr
                c = ceil(stored_nr / vector_reg_element_width)
                if (remained_nr >= 2 * vector_reg_element_width):
                    inst = current_indent + f"ldp {used_vec_regs[r][c].q_reg_name}, {used_vec_regs[r][c+1].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{2 * self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c+2) * vector_reg_element_width - 1}列对应的向量寄存器"
                    if not only_support_beta_one:
                        inst += (
                                '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}列对应的向量寄存器乘以beta"
                                + '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c+1].v_reg_name}.4s, {used_vec_regs[r][c+1].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]         // C矩阵块第{r}行, 第{(c + 1) * vector_reg_element_width} ~ {(c + 2) * vector_reg_element_width - 1}列对应有的向量寄存器乘以beta"
                        )
                    remained_nr -= (2 * vector_reg_element_width)
                elif (remained_nr >= vector_reg_element_width):
                    inst = current_indent + f"ldr {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 * vector_reg_element_width - 1}列对应的向量寄存器"
                    if not only_support_beta_one:
                        inst += (
                                '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= vector_reg_element_width
                elif remained_nr == 3:
                    inst = current_indent + f"ldr {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 1 * vector_reg_element_width - 2}列对应的向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ins {used_vec_regs[r][c].v_reg_name}.s[3], {used_vec_regs[r][c].v_reg_name}.s[0]   // 用C矩阵块第{r}行, 第{c * vector_reg_element_width}列的元素填充一下，避免计算出错"
                    )
                    if not only_support_beta_one:
                        inst += (
                                '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {(c + 1) * vector_reg_element_width - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 3
                elif remained_nr == 2:
                    inst = current_indent + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ldr {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}对应的列向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                                '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 2
                elif remained_nr == 1:
                    inst = current_indent + f"movi {used_vec_regs[r][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ldr {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 装载C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width  +  1 - 1}对应的列向量寄存器"
                    )
                    if not only_support_beta_one:
                        inst += (
                                '\n'
                                + current_indent + f"fmul {used_vec_regs[r][c].v_reg_name}.4s, {used_vec_regs[r][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_beta.v_reg_name}.s[{self.register_allocation.scalar_reg_beta.v_reg_lane_idx}]             // C矩阵块第{r}行, 第{c * vector_reg_element_width} ~ {c * vector_reg_element_width  +  1 - 1}对应的列向量寄存器乘以beta"
                        )
                    remained_nr -= 1
                else:
                    raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的加载")
                load_instructions.append(inst)
                if (c % 4 == 0) and (self.params.c_prefetch != PrefetchCacheLevel.NONE.value):
                    load_instructions.append(current_indent + f"prfm PLD{self.params.c_prefetch.value.upper()}STRM, [{c_row_addr_temp_reg.reg_name}, {self.register_allocation.prefetch_reg_c.reg_name}]   // 预取C矩阵数据")
        return load_instructions


    def generate_load_c(self, load_c_macro_fmt: Template, c_row_addr_temp_regs: List[AllocatedGeneralRegister]) -> List[Dict[str, List[str] | Any]]:
        """
        生成装载C的指令
        """
        load_c_macros = list()
        alpha_beta_type = self.get_result_scale_type()
        mr_list = self.params.get_mr_list()
        for mr in mr_list:
            if len(c_row_addr_temp_regs) < mr:
                raise NotImplementedError(f"当前不支持mr={mr}大于临时寄存器数量的情况，请增加临时寄存器数量")
            nr_list = self.params.get_nr_list(mr)
            for nr in nr_list:
                load_instructions = []
                if alpha_beta_type.value in [ResultScaleType.ALPHA1_BETA0.value, ResultScaleType.ALPHA_BETA0.value]:
                    load_instructions.extend(self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs))
                    load_instructions.extend(self.__generate_load_c_with_clear(mr, nr))
                elif alpha_beta_type.value in [ResultScaleType.ALPHA1_BETA1.value, ResultScaleType.ALPHA_BETA1.value]:
                    load_instructions.extend(self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs))
                    load_instructions.extend(self.__generate_load_c_with_beta_scale(mr, nr, c_row_addr_temp_regs, only_support_beta_one=True))
                else:
                    load_instructions.extend(self.__generate_c_row_addr(mr, nr, c_row_addr_temp_regs))
                    load_instructions.extend(self.__generate_load_c_with_beta_scale(mr, nr, c_row_addr_temp_regs, only_support_beta_one=False))

                load_c_macros.append({
                    "name": load_c_macro_fmt.substitute(nr=nr, mr=mr),
                    "instructions": load_instructions,
                })
        return load_c_macros

    def generate_store_c(self, store_c_macro_fmt: Template, c_row_addr_temp_regs: List[AllocatedGeneralRegister]) -> List[Dict[str, List[str] | Any]]:
        """
        生成保存C的指令
        """
        store_c_macros = list()
        current_indent = "    "
        vector_reg_element_width = self.get_elements_per_vector_reg()
        mr_list = self.params.get_mr_list()
        for mr in mr_list:
            if len(c_row_addr_temp_regs) < mr:
                raise NotImplementedError(f"当前不支持mr={mr}大于临时寄存器数量的情况，请增加临时寄存器数量")
            nr_list = self.params.get_nr_list(mr)
            for nr in nr_list:
                rows, cols = mr, ceil(nr / vector_reg_element_width)
                used_vec_regs, idle_vec_regs = self.register_allocation.c_vector_regs.get_rearranged_reg_names(rows, cols)
                store_instructions = []
                store_instructions.append(current_indent + f"// 保存C矩阵块: mr={mr}, nr={nr}")

                store_instructions.append(current_indent + "// 计算每行的地址")
                for r in range(rows):
                    store_instructions.append(current_indent + f"// 计算C矩阵第{r}行地址并保存")
                    c_row_addr_temp_reg = c_row_addr_temp_regs[r]
                    if r == 0:
                        store_instructions.append(current_indent + f"mov {c_row_addr_temp_reg.reg_name}, {self.register_allocation.pointer_reg_c_for_current_block_base.reg_name}   // 计算C矩阵第{r}行地址")
                    else:
                        last_c_row_addr_temp_reg = c_row_addr_temp_regs[r - 1]
                        store_instructions.append(current_indent + f"add {c_row_addr_temp_reg.reg_name}, {last_c_row_addr_temp_reg.reg_name}, {self.register_allocation.stride_reg_ldc.reg_name}   // 计算C矩阵第{r}行地址")
                store_instructions.append(current_indent + "// 开始保存矩阵C块的数据到内存")
                for r in range(rows):
                    c_row_addr_temp_reg = c_row_addr_temp_regs[r]
                    remained_nr = nr
                    while remained_nr > 0:
                        stored_nr = nr - remained_nr
                        c = ceil(stored_nr / vector_reg_element_width)
                        # 计算剩余的元素数
                        if (remained_nr >= 2 * vector_reg_element_width):
                            store_instructions.append(current_indent + f"stp {used_vec_regs[r][c].q_reg_name}, {used_vec_regs[r][c+1].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{2 * self._get_vector_byte_width()}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width} ~ {c*vector_reg_element_width + 2 * vector_reg_element_width - 1}列对应的向量寄存器")
                            remained_nr -= (2 * vector_reg_element_width)
                        elif remained_nr >= vector_reg_element_width:
                            store_instructions.append(current_indent + f"str {used_vec_regs[r][c].q_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width()}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width} ~ {c * vector_reg_element_width + vector_reg_element_width - 1}列对应的向量寄存器")
                            remained_nr -= vector_reg_element_width
                        elif remained_nr == 3:
                            inst = current_indent + f"str {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}列对应的向量寄存器"
                            inst += (
                                    '\n' +
                                    current_indent
                                    + f"ins {used_vec_regs[r][c].v_reg_name}.s[0], {used_vec_regs[r][c].v_reg_name}.s[2]   // 将C矩阵块第{r}行, 第{c*vector_reg_element_width + 3}列对应的数据往前挪为后续的存储做准备"
                            )
                            inst += (
                                    '\n' +
                                    current_indent
                                    + f"str {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width + 3}列对应的向量寄存器"
                            )
                            store_instructions.append(inst)
                            remained_nr -= 3
                        elif remained_nr == 2:
                            store_instructions.append(current_indent + f"str {used_vec_regs[r][c].d_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 2}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width} ~ {c * vector_reg_element_width + 2 - 1}列对应的向量寄存器")
                            remained_nr -= 2
                        elif remained_nr == 1:
                            store_instructions.append(current_indent + f"str {used_vec_regs[r][c].s_reg_name}, [{c_row_addr_temp_reg.reg_name}], #{self._get_vector_byte_width() // 4}   // 保存C矩阵块第{r}行, 第{c*vector_reg_element_width} ~ {c * vector_reg_element_width + 1 - 1}列对应的向量寄存器")
                            remained_nr -= 1
                        else:
                            raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的保存")

                store_c_macros.append({
                    "name": store_c_macro_fmt.substitute(nr=nr, mr=mr),
                    "instructions": store_instructions,
                })
        return store_c_macros


    def __get_vector_reg_scheduler_for_kernel_compute(self, mr: int, kr: int, nr: int) -> List[List[List[AllocatedVectorRegister]]]:
        """
        获取内核计算体的向量寄存器调度器
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        # 先分配部分和使用的向量寄存器
        partial_sum_rows, partial_sum_cols = mr, ceil(nr / elements_per_vector_reg)
        self.register_allocation.partial_sum_vector_regs.clear()
        partial_sum_vec_allocator = self.register_allocation.partial_sum_vector_regs

        available_row_vector_num = floor(len(self.register_allocation.partial_sum_vector_regs) / (kr * partial_sum_rows))
        if partial_sum_cols <= available_row_vector_num:
            # 每行使用的部分和向量寄存器数
            per_row_used_partial_sum_vec_reg_nums = partial_sum_cols
        else:
            # 不够的话需要行内重用
            per_row_used_partial_sum_vec_reg_nums = available_row_vector_num

        # 每行使用的部分和向量寄存器数
        partial_sum_vec_scheduler = []
        for idx in range(kr):
            vec_scheduler = []
            for r in range(partial_sum_rows):
                current_row_vec_allocator = AllocatedVectorRegisters(
                    var_name = f"partial_sum[{idx}][{r}]",
                    reg_names = [next(partial_sum_vec_allocator).v_reg_name for _ in range(per_row_used_partial_sum_vec_reg_nums)],
                    precision=self.params.data_type
                )
                current_row_vec_allocator.set_infinite_iteration(True)
                vec_scheduler.append([next(current_row_vec_allocator) for _ in range(partial_sum_cols)])
            partial_sum_vec_scheduler.append(vec_scheduler)
        return partial_sum_vec_scheduler


    def __generate_partial_sum_addr(self, mr:int, kr: int, partial_sum_addr_regs: List[AllocatedGeneralRegister], offset_regs: List[AllocatedGeneralRegister], tmp_reg: AllocatedGeneralRegister, with_prfm: bool = False) -> List[str]:
        """
        生成计算部分和地址和偏移的指令
        """
        instructions = []
        current_indent = "    "
        for m in range(mr):
            instructions.append(current_indent + f"// 计算部分和第{m}行的偏移地址")
            offset_reg = offset_regs[m]
            if m == 0:
                instructions.append(current_indent + f"mov {offset_reg.reg_name}, {self.register_allocation.offset_reg_partial_sum_for_current_block.reg_name}   // 获取部分和第{m}行的偏移地址")
            else:
                last_offset_reg = offset_regs[m - 1]
                instructions.append(current_indent + f"add {offset_reg.reg_name}, {last_offset_reg.reg_name}, {self.register_allocation.stride_reg_ldp.reg_name}   // 计算部分和第{m}行的偏移地址")

        instructions.append(current_indent + f"// 获取当前计算逻辑的部分和首地址: 共{kr}个")
        for idx in range(kr):
            instructions.append(current_indent + f"// 获取第{idx}个部分和的首地址")
            partial_sum_addr_reg = partial_sum_addr_regs[idx]
            if idx == 0:
                instructions.append(
                    current_indent + f"mov {tmp_reg.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}  // 获取已处理的K维度计数器用于标识当前块的开始位置"
                    + '\n' +
                    current_indent + f"lsl {tmp_reg.reg_name}, {tmp_reg.reg_name}, #3   // 计算已处理K维度对应的字节偏移量, 64位系统的指针偏移量是8字节"
                    + '\n' +
                    current_indent + f"add {tmp_reg.reg_name}, {self.register_allocation.pointer_reg_partial_sum.reg_name}, {tmp_reg.reg_name}   // 计算当前块的部分和基地址"
                    + '\n' +
                    current_indent + f"ldr {partial_sum_addr_reg.reg_name}, [{tmp_reg.reg_name}], #8   // 获取当前块第{idx}个部分和的首地址, 并移动到下一个指针"
                )
            else:
                instructions.append(current_indent + f"ldr {partial_sum_addr_reg.reg_name}, [{tmp_reg.reg_name}], #8   // 计算第{idx}个部分和的首地址, 并移动到下一个指针")
            if with_prfm and (self.params.partial_sum_prefetch != PrefetchCacheLevel.NONE.value):
                for m in range(mr):
                    offset_reg = offset_regs[m]
                    instructions.append(current_indent + f"prfm PLD{self.params.partial_sum_prefetch.value.upper()}STRM, [{partial_sum_addr_reg.reg_name}, {offset_reg.reg_name}]   // 预取第{idx}个部分和第{m}行的数据")
        return instructions


    def __load_mat_partial_sum_data_in_kernel_compute_with_alpha_scale(self, mr: int, kr: int, nr: int,
                                                             partial_sum_addr_regs: List[AllocatedGeneralRegister], partial_sum_row_offset_regs: List[AllocatedGeneralRegister],
                                                             partial_sum_vec_regs_scheduler: List[List[List[AllocatedVectorRegister]]],
                                                            prfm_tmp_reg: AllocatedGeneralRegister, round_idx: int) -> List[str]:
        """
        生成内核计算体中装载矩阵A数据的指令
        """
        if round_idx >= len(partial_sum_vec_regs_scheduler[0][0]):
            raise ValueError(f"round_idx={round_idx} 超出部分和向量寄存器调度器范围{len(partial_sum_vec_regs_scheduler[0][0])}")
        instructions = []
        current_indent = "    "
        alpha_beta_type = self.get_result_scale_type()
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        instructions.append(current_indent + f"// 装载矩阵A第{round_idx}轮的数据: mr={mr}, kr={kr}")
        remained_nr = nr - round_idx * elements_per_vector_reg

        loaded_nr = nr - remained_nr
        c = ceil(loaded_nr / elements_per_vector_reg)
        if remained_nr >= elements_per_vector_reg:
            for idx in range(kr):
                partial_sum_scheduler = partial_sum_vec_regs_scheduler[idx]
                for m in range(mr):
                    instructions.append(current_indent + f"ldr {partial_sum_scheduler[m][c].q_reg_name}, [{partial_sum_addr_regs[idx].reg_name}, {partial_sum_row_offset_regs[m].reg_name}]   // 装载第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 1}列对应的向量寄存器")
                    if alpha_beta_type.value in [ResultScaleType.ALPHA_BETA0.value, ResultScaleType.ALPHA_BETA1.value, ResultScaleType.ALPHA_BETA.value]:
                        instructions.append(
                            current_indent +
                            f"\n{current_indent}fmul {partial_sum_scheduler[m][c].v_reg_name}.4s, {partial_sum_scheduler[m][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]             // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 1}列对应的向量寄存器乘以alpha"
                        )
                    if idx == kr - 1:
                        instructions.append('\n')  # 增加一个空行，便于阅读
                        instructions.append(current_indent + f"add {partial_sum_row_offset_regs[m].reg_name}, {partial_sum_row_offset_regs[m].reg_name}, #{self._get_vector_byte_width()}   // 更新部分和第{m}行的偏移地址")
        elif remained_nr == 3:
            for idx in range(kr):
                partial_sum_scheduler = partial_sum_vec_regs_scheduler[idx]
                for m in range(mr):
                    inst = current_indent + f"ldr {partial_sum_scheduler[m][c].q_reg_name}, [{partial_sum_addr_regs[idx].reg_name}, {partial_sum_row_offset_regs[m].reg_name}]   // 装载第{idx}个部分和B第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1 * elements_per_vector_reg - 2}列对应的向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ins {partial_sum_scheduler[m][c].v_reg_name}.s[3], {partial_sum_scheduler[m][c].v_reg_name}.s[0]    // 装载第{idx}个部分和B第{m}行, 第{c * elements_per_vector_reg}列对应的元素清零向量寄存器的第3个元素, 防止计算出错"
                    )
                    instructions.append(inst)
                    if alpha_beta_type.value in [ResultScaleType.ALPHA_BETA0.value, ResultScaleType.ALPHA_BETA1.value, ResultScaleType.ALPHA_BETA.value]:
                        instructions.append(
                            current_indent +
                            f"\n{current_indent}fmul {partial_sum_scheduler[m][c].v_reg_name}.4s, {partial_sum_scheduler[m][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]             // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 2}列对应的向量寄存器乘以alpha"
                        )
                    if idx == kr - 1:
                        instructions.append('\n')  # 增加一个空行，便于阅读
                        instructions.append(current_indent + f"add {partial_sum_row_offset_regs[m].reg_name}, {partial_sum_row_offset_regs[m].reg_name}, #3   // 更新部分和第{m}行的偏移地址")
        elif remained_nr == 2:
            for idx in range(kr):
                partial_sum_scheduler = partial_sum_vec_regs_scheduler[idx]
                for m in range(mr):
                    inst = current_indent + f"movi {partial_sum_scheduler[m][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ldr {partial_sum_scheduler[m][c].d_reg_name}, [{partial_sum_addr_regs[idx].reg_name}, {partial_sum_row_offset_regs[m].reg_name}]   // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 2}列对应的向量寄存器乘以alpha"
                    )
                    instructions.append(inst)
                    if alpha_beta_type.value in [ResultScaleType.ALPHA_BETA0.value, ResultScaleType.ALPHA_BETA1.value, ResultScaleType.ALPHA_BETA.value]:
                        instructions.append(
                            current_indent +
                            f"\n{current_indent}fmul {partial_sum_scheduler[m][c].v_reg_name}.4s, {partial_sum_scheduler[m][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]             // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 2}列对应的向量寄存器乘以alpha"
                        )

                    if idx == kr - 1:
                        instructions.append('\n')  # 增加一个空行，便于阅读
                        instructions.append(current_indent + f"add {partial_sum_row_offset_regs[m].reg_name}, {partial_sum_row_offset_regs[m].reg_name}, #2   // 更新部分和第{m}行的偏移地址")
        elif remained_nr == 1:
            for idx in range(kr):
                partial_sum_scheduler = partial_sum_vec_regs_scheduler[idx]
                for m in range(mr):
                    inst = current_indent + f"movi {partial_sum_scheduler[m][c].v_reg_name}.4s, #0   // 先清零向量寄存器"
                    inst += (
                            '\n'
                            + current_indent + f"ldr {partial_sum_scheduler[m][c].s_reg_name}, [{partial_sum_addr_regs[idx].reg_name}, {partial_sum_row_offset_regs[m].reg_name}]  // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + 1}列对应的向量寄存器乘以alpha"
                    )
                    instructions.append(inst)
                    if alpha_beta_type.value in [ResultScaleType.ALPHA_BETA0.value, ResultScaleType.ALPHA_BETA1.value, ResultScaleType.ALPHA_BETA.value]:
                        instructions.append(
                            current_indent +
                            f"\n{current_indent}fmul {partial_sum_scheduler[m][c].v_reg_name}.4s, {partial_sum_scheduler[m][c].v_reg_name}.4s, {self.register_allocation.scalar_reg_alpha.v_reg_name}.s[{self.register_allocation.scalar_reg_alpha.v_reg_lane_idx}]             // 第{idx}个部分和第{m}行, 第{c * elements_per_vector_reg} ~ {c * elements_per_vector_reg + elements_per_vector_reg - 2}列对应的向量寄存器乘以alpha"
                        )

                    if idx == kr - 1:
                        instructions.append('\n')  # 增加一个空行，便于阅读
                        instructions.append(current_indent + f"add {partial_sum_row_offset_regs[m].reg_name}, {partial_sum_row_offset_regs[m].reg_name}, #1   // 更新部分和第{m}行的偏移地址")
        else:
            raise NotImplementedError(f"不支持的剩余nr={remained_nr}元素数的加载")

        if (((round_idx + 1) % 4) == 0) and (self.params.partial_sum_prefetch != PrefetchCacheLevel.NONE.value):
            for idx in range(kr):
                for m in range(mr):
                    instructions.append(
                        current_indent + f"add {prfm_tmp_reg.reg_name}, {self.register_allocation.prefetch_reg_partial_sum.reg_name}, {partial_sum_row_offset_regs[m].reg_name}   // 计算预取地址"
                    )
                    instructions.append(current_indent + f"prfm PLD{self.params.partial_sum_prefetch.value.upper()}STRM, [{partial_sum_addr_regs[idx].reg_name}, {prfm_tmp_reg.reg_name}]   // 预取部分和矩阵数据")
        return instructions

    def __update_dealt_k_for_dim_k(self, kr: int) -> List[str]:
        """
        更新已处理的K维度计数器
        """
        current_indent = "    "
        instructions = [
            current_indent + f"// 更新已处理的K维度计数器: kr={kr}",
            current_indent + f"add {self.register_allocation.loop_reg_dealt_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}, #{kr}   // 更新已处理的K维度计数器",
            ]
        return instructions

    def __compute_result_in_kernel_compute(self, mr: int, kr: int, nr: int, partial_sum_vec_regs_scheduler: List[List[List[AllocatedVectorRegister]]], round_idx: int) -> List[str]:
        """
        生成内核计算体中计算结果的指令
        """
        elements_per_vector_reg = self.get_elements_per_vector_reg()
        instructions = []
        current_indent = "    "
        instructions.append(current_indent + f"// 计算结果: mr={mr}, kr={kr}, nr={nr}, round_idx={round_idx}")
        result_rows, result_cols = mr, ceil(nr / elements_per_vector_reg)
        if round_idx >= result_cols:
            raise ValueError(f"round_idx={round_idx} 超出结果矩阵向量寄存器范围{result_cols}")
        # 计算C矩阵的对应向量寄存器
        c_vec_regs, _ = self.register_allocation.c_vector_regs.get_rearranged_reg_names(result_rows, result_cols)
        # 计算逻辑
        for idx in range(kr):
            partial_sum_scheduler = partial_sum_vec_regs_scheduler[idx]
            for m in range(mr):
                c_vec_reg = c_vec_regs[m][round_idx]
                if round_idx >= len(partial_sum_scheduler[m]):
                    raise ValueError(f"round_idx={round_idx} 超出部分和向量寄存器调度器范围{len(partial_sum_scheduler[m])}")
                partial_sum_vec_reg = partial_sum_scheduler[m][round_idx]
                instructions.append(
                    current_indent + f"fadd {c_vec_reg.v_reg_name}.4s, {c_vec_reg.v_reg_name}.4s, {partial_sum_vec_reg.v_reg_name}.4s   // 计算C矩阵块第{m}行, 第{round_idx * elements_per_vector_reg} ~ {round_idx * elements_per_vector_reg + elements_per_vector_reg - 1}列对应的向量寄存器加上第{idx}个部分和的值(可能存在无效的加法操作，在存储时会被过滤掉)"
                )
        return instructions

    def __can_arrange_partial_sum_compute_alternately(self, mr: int, nr: int, kr: int,
                                                      partial_sum_regs_scheduler: List[List[List[AllocatedVectorRegister]]],
                                                      compute_round_idx_start: int, compute_round_step: int,
                                                      load_round_idx_start: int, load_round_step: int) -> bool:
        """
        统计能否交替安排加载部分和计算指令
        """
        total_round = ceil(nr / self.get_elements_per_vector_reg())
        if (compute_round_idx_start + compute_round_step > total_round) or (load_round_idx_start + load_round_step > total_round):
            raise ValueError(f"计算轮次或加载轮次超出总轮次: total_round={total_round}, compute_round_idx_start={compute_round_idx_start}, compute_round_step={compute_round_step}, load_round_idx_start={load_round_idx_start}, load_round_step={load_round_step}")

        for idx in range(kr):
            partial_sum_scheduler = partial_sum_regs_scheduler[idx]
            for m in range(mr):
                partial_sum_vec_regs_for_compute = partial_sum_scheduler[m][compute_round_idx_start: compute_round_idx_start + compute_round_step]
                partial_sum_vec_regs_for_load = partial_sum_scheduler[m][load_round_idx_start: load_round_idx_start + load_round_step]
                for reg in partial_sum_vec_regs_for_load:
                    if reg in partial_sum_vec_regs_for_compute:
                        return False
        return True


    def generate_kernel_compute(self, kernel_macro_fmt: Template, available_temp_regs: List[AllocatedGeneralRegister]) -> List[Dict[str, List[str] | Any]]:
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
                current_indent+ f"* 内核大小: mr={mr}, nr={nr}, kr={kr}",
                ]
            partial_sum_current_block_base_addr_temp_regs = [available_temp_regs_for_loop.pop(0) for _ in range(kr)]
            kernel_instructions.extend([
                current_indent + f"* 部分和{i}当前块基地址寄存器: {partial_sum_current_block_base_addr_temp_regs[i].reg_name}" for i in range(kr)
            ])
            partial_sum_row_offset_temp_regs = [available_temp_regs_for_loop.pop(0) for _ in range(mr)]
            kernel_instructions.extend([
                current_indent + f"* 部分和当前行偏移寄存器: {partial_sum_row_offset_temp_regs[i].reg_name}" for i in range(mr)
            ])

            partial_sum_regs_scheduler = self.__get_vector_reg_scheduler_for_kernel_compute(mr, kr, nr)
            kernel_instructions.append(
                current_indent + f"* 部分和实际使用的向量寄存器: {','.join([reg.v_reg_name for reg in sorted(list(set(flatten_list(partial_sum_regs_scheduler))), key = lambda x: x.reg_id)])}"
            )
            kernel_instructions.append(current_indent + "* 部分和向量寄存器调度表:")
            for idx in range(len(partial_sum_regs_scheduler)):
                for r in range(len(partial_sum_regs_scheduler[idx])):
                    kernel_instructions.append(
                        current_indent + f"* 部分和{idx}第{r}行: " + ",".join([reg.v_reg_name for reg in partial_sum_regs_scheduler[idx][r]])
                    )

            kernel_instructions.append(
                "* 矩阵C使用的寄存器列表: "
            )
            mat_c_vec_regs, _ = self.register_allocation.c_vector_regs.get_rearranged_reg_names(mr, ceil(nr / elements_per_vector_reg))
            for r in range(len(mat_c_vec_regs)):
                kernel_instructions.append(
                    current_indent + f"* 第{r}行: " + ",".join([reg.v_reg_name for reg in mat_c_vec_regs[r]])
                )

            remained_k_reg = available_temp_regs_for_loop.pop(0)
            kernel_instructions.append(current_indent + f"* 计算剩余K维度的临时寄存器: {remained_k_reg.reg_name}")
            prfm_tmp_reg = available_temp_regs_for_loop.pop(0)
            kernel_instructions.append(current_indent + f"* 预取计算临时寄存器: {prfm_tmp_reg.reg_name}")
            current_partial_sum_addr_offset_reg = available_temp_regs_for_loop.pop(0)
            kernel_instructions.append(current_indent + f"* 当前部分和地址偏移临时寄存器: {current_partial_sum_addr_offset_reg.reg_name}")
            kernel_instructions.append("****/")

            kernel_instructions.extend(self.__generate_partial_sum_addr(mr, kr, partial_sum_current_block_base_addr_temp_regs, partial_sum_row_offset_temp_regs, current_partial_sum_addr_offset_reg, True))

            kernel_instructions.append(current_indent + f"sub {remained_k_reg.reg_name}, {self.register_allocation.dim_reg_k.reg_name}, {self.register_allocation.loop_reg_dealt_k.reg_name}   // 初始化剩余K维度计数器(即未累加的部分和数量)")

            kernel_instructions.append(f"KERNEL_COMPUTE_MR{mr}_NR{nr}_KR{kr}_INIT_{label_suffix} :    // 初始化内核计算")
            # 装载部分和的首轮数据
            load_partial_sum_first_loop_instructions = self.__load_mat_partial_sum_data_in_kernel_compute_with_alpha_scale(
                mr=mr, kr=kr, nr=nr,
                partial_sum_addr_regs=partial_sum_current_block_base_addr_temp_regs, partial_sum_row_offset_regs=partial_sum_row_offset_temp_regs,
                partial_sum_vec_regs_scheduler=partial_sum_regs_scheduler, prfm_tmp_reg=prfm_tmp_reg, round_idx=0
            )
            kernel_instructions.extend(load_partial_sum_first_loop_instructions)

            # 判断是否进入ping-pong计算
            kernel_instructions.append(current_indent + f"// 判断是否进入Ping-Pong计算: kr={kr}")
            kernel_instructions.append(current_indent + f"cmp {remained_k_reg.reg_name}, #{ 2 * kr}   // 判断是否进入Ping-Pong计算: kr={kr}")
            kernel_instructions.append(current_indent + f"blt KERNEL_COMPUTE_MR{mr}_NR{nr}_KR{kr}_FINISH_{label_suffix}    // 如果剩余K维度小于2*kr={2*kr}，跳转到最后一轮计算")

            # Ping-Pong计算
            kernel_instructions.append(f"KERNEL_COMPUTE_MR{mr}_NR{nr}_KR{kr}_PINGPONG_{label_suffix} :    // 内核计算体Ping-Pong计算开始, mr={mr}, kr={kr}, nr={nr}")
            total_compute_round = ceil(nr / elements_per_vector_reg)
            compute_round_list = [i for i in range(0, total_compute_round)]
            load_round_list = compute_round_list[1:] + compute_round_list[:1]
            for (compute_round_idx, load_round_idx) in zip(compute_round_list, load_round_list):
                if load_round_idx == 0:
                    kernel_instructions.extend(self.__update_dealt_k_for_dim_k(kr))
                    kernel_instructions.extend(self.__generate_partial_sum_addr(mr, kr, partial_sum_current_block_base_addr_temp_regs, partial_sum_row_offset_temp_regs, current_partial_sum_addr_offset_reg, True))

                compute_instructions = self.__compute_result_in_kernel_compute(mr, kr, nr, partial_sum_regs_scheduler, compute_round_idx)
                load_partial_sum_loop_instructions = self.__load_mat_partial_sum_data_in_kernel_compute_with_alpha_scale(
                    mr=mr, kr=kr, nr=nr,
                    partial_sum_addr_regs=partial_sum_current_block_base_addr_temp_regs, partial_sum_row_offset_regs=partial_sum_row_offset_temp_regs,
                    partial_sum_vec_regs_scheduler=partial_sum_regs_scheduler, prfm_tmp_reg=prfm_tmp_reg, round_idx=load_round_idx
                )

                can_arrange_load_compute_alternately = self.__can_arrange_partial_sum_compute_alternately(
                    mr=mr, nr=nr, kr=kr,
                    partial_sum_regs_scheduler=partial_sum_regs_scheduler,
                    compute_round_idx_start=compute_round_idx, compute_round_step=1,
                    load_round_idx_start=load_round_idx, load_round_step=1
                )

                if can_arrange_load_compute_alternately is False:  # 提前预取时一定要保证当前轮计算先完成
                    kernel_instructions.extend(compute_instructions)
                    kernel_instructions.extend(load_partial_sum_loop_instructions)
                else:
                    for inst in zip_longest(compute_instructions, load_partial_sum_loop_instructions, fillvalue=None):
                        for i in inst:
                            if i is not None:
                                kernel_instructions.append(i)

                if compute_round_idx == (total_compute_round - 1):
                    # 更新剩余K维度计数器
                    kernel_instructions.append(current_indent + f"sub {remained_k_reg.reg_name}, {remained_k_reg.reg_name}, #{kr}   // 更新剩余K维度计数器")
                    # 判断是否继续Ping-Pong计算
                    kernel_instructions.append(current_indent + f"cmp {remained_k_reg.reg_name}, #{2 * kr}   // 判断是否继续Ping-Pong计算: kr={kr}")
                    kernel_instructions.append(current_indent + f"bge KERNEL_COMPUTE_MR{mr}_NR{nr}_KR{kr}_PINGPONG_{label_suffix}    // 如果剩余K维度大于等于2*kr={2*kr}，继续Ping-Pong计算")

            # 最后一轮计算
            kernel_instructions.append(f"KERNEL_COMPUTE_MR{mr}_NR{nr}_KR{kr}_FINISH_{label_suffix} :    // 最后一轮计算开始, mr={mr}, kr={kr}, nr={nr}")
            for (compute_round_idx, load_round_idx) in zip(compute_round_list, load_round_list):

                compute_instructions = self.__compute_result_in_kernel_compute(mr, kr, nr, partial_sum_regs_scheduler, compute_round_idx)

                if load_round_idx > 0:
                    load_partial_sum_loop_instructions = self.__load_mat_partial_sum_data_in_kernel_compute_with_alpha_scale(
                        mr=mr, kr=kr, nr=nr,
                        partial_sum_addr_regs=partial_sum_current_block_base_addr_temp_regs, partial_sum_row_offset_regs=partial_sum_row_offset_temp_regs,
                        partial_sum_vec_regs_scheduler=partial_sum_regs_scheduler, prfm_tmp_reg=prfm_tmp_reg, round_idx=load_round_idx
                    )
                    can_arrange_load_compute_alternately = self.__can_arrange_partial_sum_compute_alternately(
                        mr=mr, nr=nr, kr=kr,
                        partial_sum_regs_scheduler=partial_sum_regs_scheduler,
                        compute_round_idx_start=compute_round_idx, compute_round_step=1,
                        load_round_idx_start=load_round_idx, load_round_step=1
                    )
                else:
                    load_partial_sum_loop_instructions = []
                    can_arrange_load_compute_alternately = False

                if can_arrange_load_compute_alternately is False:  # 提前预取时一定要保证当前轮计算先完成
                    kernel_instructions.extend(compute_instructions)
                    kernel_instructions.extend(load_partial_sum_loop_instructions)
                else:
                    for inst in zip_longest(compute_instructions, load_partial_sum_loop_instructions, fillvalue=None):
                        for i in inst:
                            if i is not None:
                                kernel_instructions.append(i)

                if compute_round_idx == (total_compute_round - 1):
                    kernel_instructions.extend(self.__update_dealt_k_for_dim_k(kr))

            kernel_compute_macros.append({
                "name": kernel_macro_fmt.substitute(mr=mr, kr=kr, nr=nr),
                "instructions": kernel_instructions
            })

        return kernel_compute_macros
