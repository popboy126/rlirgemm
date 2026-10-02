# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 23:08
# Contact       : hucheng.lew@qq.com
# File          : register_allocation.py
# IDE           : PyCharm
# Description   :

from dataclasses import dataclass
from typing import List, Tuple

from dataclasses import dataclass
from typing import List, Tuple, Optional
from copy import deepcopy
from gemm_codegen.common import ElementPrecision

@dataclass
class AllocatedGeneralRegister:
    """通用寄存器的分配信息"""
    var_name: str
    reg_name: str


class AllocatedGeneralRegisters:
    """通用寄存器分配结果"""
    def __init__(self, regs: List[AllocatedGeneralRegister]):
        """
        初始化
        """
        self.regs: List[AllocatedGeneralRegister] = deepcopy(regs)
        self._iter_index: int = 0  # 迭代位置
        self._infinite_iter: bool = False  # 是否无限迭代

    def __iter__(self):
        self._iter_index = 0  # 重置迭代位置
        return self

    def __next__(self):
        if not self._infinite_iter:
            if self._iter_index < len(self.regs):
                field_name = self.regs[self._iter_index]
                self._iter_index += 1
                return field_name
            else:
                raise StopIteration
        else:
            if self._iter_index >= len(self.regs):
                self._iter_index = 0  # 重置为开头
            field_name = self.regs[self._iter_index]
            self._iter_index += 1
            return field_name

    def __len__(self):
        return len(self.regs)

    def set_infinite_iteration(self, infinite: bool):
        """设置是否无限迭代"""
        self._infinite_iter = infinite

    def __getitem__(self, index):
        return self.regs[index]

    def clear(self) -> None:
        """清空状态"""
        self._iter_index = 0
        self._infinite_iter = False

class AllocatedVectorRegister:
    """向量寄存器的分配信息"""
    def __init__(self, var_name: str, reg_name: str, precision: ElementPrecision, position: Optional[int] = None):
        """
        初始化
        """
        if reg_name.isdigit():
            self._reg_id: int = int(reg_name)
        else:
            reg_id = reg_name.lstrip('v')
            if not reg_id.isdigit():
                raise ValueError(f"无效的向量寄存器名称: {reg_name}")
            self._reg_id: int = int(reg_id)
        self._var_name: str = var_name
        self._precision: ElementPrecision = precision
        self._position: Optional[int] = position

    @property
    def v_reg_name(self) -> str:
        """获取向量寄存器名称"""
        return f"v{self._reg_id}"

    @property
    def v_reg_name_with_precision(self) -> str:
        """获取带精度后缀的向量寄存器名称"""
        if self._precision.precision == self._precision.precision.FP32:
            if self._position is not None:
                return f"{self.v_reg_name}.s[{self._position}]"
            else:
                return f"{self.v_reg_name}.4s"
        else:
            raise NotImplementedError("当前仅支持FP32精度的向量寄存器名称后缀")

    @property
    def v_reg_lane_idx(self) -> int:
        """获取向量寄存器的lane索引"""
        if self._position is not None:
            return self._position
        else:
            raise ValueError("寄存器位置未指定，无法获取lane索引")

    @property
    def q_reg_name(self) -> str:
        """获取128-bit寄存器名称"""
        return f"q{self._reg_id}"

    @property
    def d_reg_name(self) -> str:
        """获取64-bit寄存器名称"""
        return f"d{self._reg_id}"

    @property
    def s_reg_name(self) -> str:
        """获取32-bit寄存器名称"""
        return f"s{self._reg_id}"

    @property
    def var_name(self) -> str:
        """获取变量名称"""
        return self._var_name

    @property
    def position(self) -> Optional[str]:
        """获取寄存器位置描述"""
        return self._position

    @property
    def reg_id(self) -> int:
        """获取寄存器ID"""
        return self._reg_id

class AllocatedVectorRegisters:
    """向量寄存器分配结果"""
    def __init__(self, var_name: str, reg_names: List[str], precision: ElementPrecision):
        """
        初始化
        """
        self.var_name: str = var_name
        self.regs: List[AllocatedVectorRegister] = [AllocatedVectorRegister(var_name, reg, precision) for reg in reg_names]
        self._iter_index: int = 0  # 迭代位置
        self._infinite_iter: bool = False  # 是否无限迭代

    def __iter__(self):
        self._iter_index = 0  # 重置迭代位置
        return self

    def __next__(self):
        if not self._infinite_iter:
            if self._iter_index < len(self.regs):
                reg = self.regs[self._iter_index]
                self._iter_index += 1
                return reg
            raise StopIteration
        else:
            if self._iter_index >= len(self.regs):
                self._iter_index = 0  # 重置为开头
            reg = self.regs[self._iter_index]
            self._iter_index += 1
            return reg

    def __len__(self):
        return len(self.regs)

    def set_infinite_iteration(self, infinite: bool):
        """设置是否无限迭代"""
        self._infinite_iter = infinite

    def get_reg_name_str(self) -> str:
        """
        获取寄存器名称字符串
        """
        return ", ".join([ reg.v_reg_name for reg in self.regs])

    def get_var_name(self) -> str:
        """获取变量名称"""
        return self.var_name

    def clear(self) -> None:
        """清空状态"""
        self._iter_index = 0
        self._infinite_iter = False

    def get_rearranged_reg_names(self, rows: int, cols: int) -> Tuple[List[List[AllocatedVectorRegister]], List[AllocatedVectorRegister]]:
        """
        获取重新排列的寄存器列表
        返回值: (重排的二维寄存器列表, 剩余的一维寄存器列表)
        """
        total_needed = rows * cols
        if total_needed > len(self.regs):
            raise ValueError("寄存器数量不足以重排为指定的行列数")

        rearranged_regs = []
        for r in range(rows):
            row_regs = []
            for c in range(cols):
                row_regs.append(self.regs[r * cols + c])
            rearranged_regs.append(row_regs)
        remaining_regs = self.regs[total_needed:]
        return rearranged_regs, remaining_regs


@dataclass
class RegisterAllocation:
    """寄存器分配信息"""
    # 向量寄存器分配
    c_vector_regs: AllocatedVectorRegisters  # C矩阵寄存器
    partial_sum_vector_regs: AllocatedVectorRegisters  # 部分和矩阵寄存器

    # 通用寄存器分配
    pointer_reg_c: AllocatedGeneralRegister  # 函数传参的矩阵C的寄存器
    stride_reg_ldc: AllocatedGeneralRegister  # 矩阵C的行/列步长寄存器

    pointer_reg_partial_sum: AllocatedGeneralRegister  # 函数传参的部分和的寄存器
    stride_reg_ldp: AllocatedGeneralRegister  # 部分和的行/列步长寄存器
    dim_reg_m: AllocatedGeneralRegister # M维度寄存器
    dim_reg_n: AllocatedGeneralRegister # N维度寄存器
    dim_reg_k: AllocatedGeneralRegister # K维度寄存器(部分和数量)

    scalar_reg_alpha: AllocatedVectorRegister  # alpha向量寄存器
    scalar_reg_beta: AllocatedVectorRegister  # beta向量寄存器

    prefetch_reg_c: AllocatedGeneralRegister  # 矩阵C预取步长寄存器
    prefetch_reg_partial_sum: AllocatedGeneralRegister  # 部分和预取步长寄存器

    loop_reg_dealt_m: AllocatedGeneralRegister  # M维度处理完后的行数寄存器
    loop_reg_dealt_n: AllocatedGeneralRegister  # N维度处理完后的行数寄存器
    loop_reg_dealt_k: AllocatedGeneralRegister  # K维度处理完后的行数寄存器

    pointer_reg_c_row_base: AllocatedGeneralRegister  # 循环M基地址寄存器
    row_offset_partial_sum_base: AllocatedGeneralRegister  # 循环M的部分和偏移寄存器

    pointer_reg_c_for_current_block_base: AllocatedGeneralRegister  # 当前块矩阵C基地址寄存器
    offset_reg_partial_sum_for_current_block: AllocatedGeneralRegister  # 当前块部分和偏移寄存器

    # 临时寄存器分配
    temp_regs: AllocatedGeneralRegisters

    def clear(self) -> None:
        """
        清空状态
        """
        self.c_vector_regs.clear()
        self.partial_sum_vector_regs.clear()
        self.temp_regs.clear()


    def get_register_allocation_report(self) -> List[Tuple[str, str]]:
        """获取寄存器分配报告"""
        report = [
            ("* 向量寄存器分配情况", f"共计: 矩阵C({len(self.c_vector_regs)}), 部分和({len(self.partial_sum_vector_regs)})"),
            (self.c_vector_regs.get_var_name(), self.c_vector_regs.get_reg_name_str()),
            (self.partial_sum_vector_regs.get_var_name(), self.partial_sum_vector_regs.get_reg_name_str()),
            ("* 通用寄存器分配情况", "用于存储地址和计算过程中的逻辑信息"),
            (self.pointer_reg_c.var_name, self.pointer_reg_c.reg_name),
            (self.stride_reg_ldc.var_name, self.stride_reg_ldc.reg_name),
            (self.pointer_reg_partial_sum.var_name, self.pointer_reg_partial_sum.reg_name),
            (self.stride_reg_ldp.var_name, self.stride_reg_ldp.reg_name),
            (self.dim_reg_m.var_name, self.dim_reg_m.reg_name),
            (self.dim_reg_n.var_name, self.dim_reg_n.reg_name),
            (self.dim_reg_k.var_name, self.dim_reg_k.reg_name),

            (self.scalar_reg_alpha.var_name, self.scalar_reg_alpha.v_reg_name_with_precision),
            (self.scalar_reg_beta.var_name, self.scalar_reg_beta.v_reg_name_with_precision),

            (self.prefetch_reg_c.var_name, self.prefetch_reg_c.reg_name),
            (self.prefetch_reg_partial_sum.var_name, self.prefetch_reg_partial_sum.reg_name),

            (self.loop_reg_dealt_m.var_name, self.loop_reg_dealt_m.reg_name),
            (self.loop_reg_dealt_n.var_name, self.loop_reg_dealt_n.reg_name),
            (self.loop_reg_dealt_k.var_name, self.loop_reg_dealt_k.reg_name),

            (self.pointer_reg_c_row_base.var_name, self.pointer_reg_c_row_base.reg_name),
            (self.row_offset_partial_sum_base.var_name, self.row_offset_partial_sum_base.reg_name),

            (self.pointer_reg_c_for_current_block_base.var_name, self.pointer_reg_c_for_current_block_base.reg_name),
            (self.offset_reg_partial_sum_for_current_block.var_name, self.offset_reg_partial_sum_for_current_block.reg_name),
        ]

        report.append(("* 临时寄存器", f"共计: {len(self.temp_regs)}"))
        for temp in self.temp_regs:
            report.append( (temp.var_name, temp.reg_name) )

        return report
