# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/11 21:29
# Contact       : hucheng.lew@qq.com
# File          : allocated_registers.py
# IDE           : PyCharm
# Description   : 寄存器分配数据对象


from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from gemm_codegen.common import ElementPrecision


@dataclass
class AllocatedGeneralRegister:
    """通用寄存器的分配信息"""

    var_name: str
    reg_name: str
    spilled: bool = False
    offset: int = 0  # 栈偏移量，用于寄存器不足时存储在栈上

    def __eq__(self, other):
        if not isinstance(other, AllocatedGeneralRegister):
            return False
        return (
            (self.reg_name == other.reg_name)
            and (self.spilled == other.spilled)
            and (self.offset == other.offset)
        )

    def __hash__(self):
        # 使用不可变且唯一的属性生成哈希值
        return hash(f"{self.reg_name}_{self.spilled}_{self.offset}")

    def __str__(self):
        """用户友好的字符串表示"""
        if self.spilled:
            return f"{self.var_name} --> [{self.reg_name} + {self.offset}] (stack)"
        else:
            return f"{self.var_name} --> {self.reg_name}"

    def __repr__(self):
        """明确的字符串表示，理想情况下能重新创建对象"""
        if self.spilled:
            return f"AllocatedGeneralRegister(var_name='{self.var_name}', reg_name='{self.reg_name}', spilled={self.spilled}, offset={self.offset})"
        else:
            return f"AllocatedGeneralRegister(var_name='{self.var_name}', reg_name='{self.reg_name}')"

    def get_reg_name(self) -> str:
        """
        获取寄存器名
        """
        if self.spilled:
            return f"[{self.reg_name} + {self.offset}] (stack)"
        else:
            return self.reg_name



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

    def __init__(
        self,
        var_name: str,
        reg_name: str,
        precision: ElementPrecision,
        position: Optional[int] = None,
    ):
        """
        初始化

        支持多种寄存器命名方式：
        - ARM NEON: "v0", "v1", ..., "v31"
        - x86 AVX: "ymm0", "ymm1", ..., "ymm15"
        - x86 SSE: "xmm0", "xmm1", ..., "xmm15"
        - 纯数字: "0", "1", ...
        """
        if reg_name.isdigit():
            self._reg_id: int = int(reg_name)
        else:
            # 支持多种前缀: v, ymm, xmm, q, d, s
            reg_id = reg_name
            for prefix in ["ymm", "xmm", "v", "q", "d", "s"]:
                if reg_name.startswith(prefix):
                    reg_id = reg_name[len(prefix) :]
                    break

            if not reg_id.isdigit():
                raise ValueError(f"无效的向量寄存器名称: {reg_name}")
            self._reg_id: int = int(reg_id)
        self._var_name: str = var_name
        self._precision: ElementPrecision = precision
        self._position: Optional[int] = position

    def __eq__(self, other):
        if not isinstance(other, AllocatedVectorRegister):
            return False
        return self.reg_id == other.reg_id

    def __hash__(self):
        # 使用不可变且唯一的属性生成哈希值
        return hash(self.reg_id)

    def __str__(self):
        """用户友好的字符串表示"""
        return f"{self.var_name} --> {self.v_reg_name}"

    def __repr__(self):
        """明确的字符串表示，理想情况下能重新创建对象"""
        return f"AllocatedVectorRegister(_var_name='{self._var_name}', _reg_id={self._reg_id})"

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
    def ymm_reg_name(self) -> str:
        """获取AVX2 256-bit寄存器名称"""
        return f"ymm{self._reg_id}"

    @property
    def xmm_reg_name(self) -> str:
        """获取128-bit寄存器名称（用于AVX/SSE）"""
        return f"xmm{self._reg_id}"

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

    def __init__(
        self, mat_name: str, reg_names: List[str], precision: ElementPrecision
    ):
        """
        初始化
        """
        self.mat_name: str = mat_name
        self.regs: List[AllocatedVectorRegister] = [
            AllocatedVectorRegister(mat_name, reg, precision) for reg in reg_names
        ]
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
        return ", ".join([reg.v_reg_name for reg in self.regs])

    def get_mat_name(self) -> str:
        """获取矩阵名称"""
        return self.mat_name

    def clear(self) -> None:
        """清空状态"""
        self._iter_index = 0
        self._infinite_iter = False

    def get_rearranged_reg_names(
        self, rows: int, cols: int
    ) -> Tuple[List[List[AllocatedVectorRegister]], List[AllocatedVectorRegister]]:
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
    """寄存器分配结果"""

    # 向量寄存器分配
    c_vector_regs: AllocatedVectorRegisters  # C矩阵寄存器
    a_vector_regs: AllocatedVectorRegisters  # A矩阵寄存器
    b_vector_regs: AllocatedVectorRegisters  # B矩阵寄存器

    # 通用寄存器分配
    pointer_reg_c: AllocatedGeneralRegister  # 函数传参的矩阵C的寄存器
    pointer_reg_a: AllocatedGeneralRegister  # 函数传参的矩阵A的寄存器
    pointer_reg_b: AllocatedGeneralRegister  # 函数传参的矩阵B的寄存器
    dim_reg_m: AllocatedGeneralRegister  # M维度寄存器
    dim_reg_n: AllocatedGeneralRegister  # N维度寄存器
    dim_reg_k: AllocatedGeneralRegister  # K维度寄存器
    stride_reg_ldc: AllocatedGeneralRegister  # 矩阵C的行/列步长寄存器
    stride_reg_lda: AllocatedGeneralRegister  # 矩阵A的行/列步长寄存器
    stride_reg_ldb: AllocatedGeneralRegister  # 矩阵B的行/列步长寄存器
    prefetch_reg_c: AllocatedGeneralRegister  # 矩阵C预取步长寄存器
    prefetch_reg_a: AllocatedGeneralRegister  # 矩阵A预取步长寄存器
    prefetch_reg_b: AllocatedGeneralRegister  # 矩阵B预取步长寄存器
    scalar_reg_alpha: AllocatedVectorRegister  # alpha向量寄存器
    scalar_reg_beta: AllocatedVectorRegister  # beta向量寄存器

    pointer_reg_c_for_loop_n: AllocatedGeneralRegister  # N循环下C的首地址指针寄存器

    pointer_reg_c_for_current_block_base: (
        AllocatedGeneralRegister  # 矩阵C的当前块的基地址指针寄存器
    )
    pointer_reg_a_for_current_block_base: (
        AllocatedGeneralRegister  # 矩阵A的当前块的基地址指针寄存器
    )
    pointer_reg_b_for_current_block_base: (
        AllocatedGeneralRegister  # 矩阵B当前的块基地址指针寄存器
    )

    loop_reg_dealt_m: AllocatedGeneralRegister  # M维度处理完后的行数寄存器
    loop_reg_dealt_n: AllocatedGeneralRegister  # N维度处理完后的行数寄存器
    loop_reg_dealt_k: AllocatedGeneralRegister  # K维度处理完后的行数寄存器

    temp_regs: AllocatedGeneralRegisters

    pointer_reg_bc: Optional[AllocatedGeneralRegister] = None  # BC打包串的指针寄存器
    pointer_reg_bc_for_loop_k: Optional[AllocatedGeneralRegister] = (
        None  # K循环下BC打包串的指针寄存器
    )

    def clear(self) -> None:
        """
        清空状态
        """
        self.c_vector_regs.clear()
        self.a_vector_regs.clear()
        self.b_vector_regs.clear()
        self.temp_regs.clear()

    def get_register_allocation_report(self) -> List[Tuple[str, str]]:
        """获取寄存器分配报告"""
        report = [
            (
                "* 向量寄存器分配情况",
                f"共计: 矩阵C({len(self.c_vector_regs)}), 矩阵A({len(self.a_vector_regs)}), 矩阵B({len(self.b_vector_regs)})",
            ),
            (self.c_vector_regs.get_mat_name(), self.c_vector_regs.get_reg_name_str()),
            (self.a_vector_regs.get_mat_name(), self.a_vector_regs.get_reg_name_str()),
            (self.b_vector_regs.get_mat_name(), self.b_vector_regs.get_reg_name_str()),
            ("* 通用寄存器分配情况", "用于存储地址和计算过程中的逻辑信息"),
            (self.pointer_reg_c.var_name, self.pointer_reg_c.get_reg_name()),
            (self.pointer_reg_a.var_name, self.pointer_reg_a.get_reg_name()),
            (self.pointer_reg_b.var_name, self.pointer_reg_b.get_reg_name()),
            (self.dim_reg_m.var_name, self.dim_reg_m.get_reg_name()),
            (self.dim_reg_n.var_name, self.dim_reg_n.get_reg_name()),
            (self.dim_reg_k.var_name, self.dim_reg_k.get_reg_name()),
            (self.stride_reg_ldc.var_name, self.stride_reg_ldc.get_reg_name()),
            (self.stride_reg_lda.var_name, self.stride_reg_lda.get_reg_name()),
            (self.stride_reg_ldb.var_name, self.stride_reg_ldb.get_reg_name()),
            (self.prefetch_reg_c.var_name, self.prefetch_reg_c.get_reg_name()),
            (self.prefetch_reg_a.var_name, self.prefetch_reg_a.get_reg_name()),
            (self.prefetch_reg_b.var_name, self.prefetch_reg_b.get_reg_name()),
            (
                self.scalar_reg_alpha.var_name,
                self.scalar_reg_alpha.v_reg_name_with_precision,
            ),
            (
                self.scalar_reg_beta.var_name,
                self.scalar_reg_beta.v_reg_name_with_precision,
            ),
            (
                self.pointer_reg_c_for_loop_n.var_name,
                self.pointer_reg_c_for_loop_n.get_reg_name(),
            ),
            (
                self.pointer_reg_c_for_current_block_base.var_name,
                self.pointer_reg_c_for_current_block_base.get_reg_name(),
            ),
            (
                self.pointer_reg_a_for_current_block_base.var_name,
                self.pointer_reg_a_for_current_block_base.get_reg_name(),
            ),
            (
                self.pointer_reg_b_for_current_block_base.var_name,
                self.pointer_reg_b_for_current_block_base.get_reg_name(),
            ),
            (self.loop_reg_dealt_m.var_name, self.loop_reg_dealt_m.get_reg_name()),
            (self.loop_reg_dealt_n.var_name, self.loop_reg_dealt_n.get_reg_name()),
            (self.loop_reg_dealt_k.var_name, self.loop_reg_dealt_k.get_reg_name()),
        ]

        if self.pointer_reg_bc:
            report.append((self.pointer_reg_bc.var_name, self.pointer_reg_bc.get_reg_name()))

        if self.pointer_reg_bc_for_loop_k:
            report.append(
                (
                    self.pointer_reg_bc_for_loop_k.var_name,
                    self.pointer_reg_bc_for_loop_k.get_reg_name(),
                )
            )

        # 临时寄存器

        report.append(("* 临时寄存器", f"共计: {len(self.temp_regs)}"))
        for temp in self.temp_regs:
            report.append((temp.var_name, temp.get_reg_name()))

        return report
