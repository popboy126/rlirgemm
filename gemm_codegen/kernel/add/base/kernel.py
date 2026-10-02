# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:59
# Contact       : hucheng.lew@qq.com
# File          : kernel.py
# IDE           : PyCharm
# Description   : 跟内核定义有关的数据结果

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from gemm_codegen.common import (
    ElementPrecision, MatrixLayout, PrefetchCacheLevel
)

@dataclass
class KernelShape:
    """内核形状"""
    mr: int
    kr: int
    nr: int
    edge_shape: bool = False  # 是否为边界形状

    def __str__(self):
        return f"<mr={self.mr}, kr={self.kr}, nr={self.nr}>"

class KernelShapes:
    """内核形状配置"""
    def __init__(self, mr: int, kr: int, nr: int):
        """
        初始化
        Args:
            mr: M维度分块大小
            kr: K维度分块大小
            nr: N维度分块大小
        """
        self._mr = mr
        self._kr = kr
        self._nr = nr
        self._edge_shapes: List[KernelShape] = []

    @property
    def mr(self) -> int:
        """获取M维度分块大小"""
        return self._mr

    @property
    def kr(self) -> int:
        """获取K维度分块大小"""
        return self._kr

    @property
    def nr(self) -> int:
        """获取N维度分块大小"""
        return self._nr

    @property
    def edge_kernel_shapes(self) -> Optional[List[KernelShape]]:
        """获取边界内核形状列表"""
        return self._edge_shapes if self._edge_shapes else None

    def __len__(self):
        return 1 + len(self._edge_shapes)

    def set_edge_shapes(self, edge_shapes: List[Tuple[int, int, int]]):
        """设置边界内核形状列表

        Args:
            edge_shapes: 边界内核形状列表，每个元素为(mr, kr, nr)元组
        """
        self._edge_shapes = [KernelShape(mr, kr, nr, edge_shape=True) for (mr, kr, nr) in edge_shapes]

    def get_edge_shapes_str(self) -> str:
        """获取边界内核形状字符串表示"""
        return ", ".join([str(shape) for shape in self._edge_shapes])

    def get_kernel_shapes_by_nr(self, nr: int) -> List[KernelShape]:
        """根据nr获取对应的内核形状列表"""
        shapes = []
        if self._nr == nr:
            shapes.append(KernelShape(self._mr, self._kr, self._nr))
        for shape in self._edge_shapes:
            if shape.nr == nr:
                shapes.append(shape)
        return shapes


@dataclass
class KernelParameters:
    """内核参数配置"""
    # 基本配置
    platform: str  # 平台架构: AMD64, X86, ARMv8, ARMv9, RISC-V
    instruction_set: str  # 指令集: AVX2, AVX512, NEON, SVE, SVE2, RVV等
    data_type: ElementPrecision  # 数据类型: f32, f64, i8, i16, i32, i64

    # 分块尺寸
    kernel_shapes: KernelShapes

    # 矩阵布局和转置
    layout: MatrixLayout = MatrixLayout.ROW_MAJOR

    # 访存优化
    c_prefetch: PrefetchCacheLevel = PrefetchCacheLevel.NONE
    partial_sum_prefetch: PrefetchCacheLevel = PrefetchCacheLevel.NONE

    # 寄存器优化
    vec_reg_reuse_interval: int = 3  # 寄存器重用间隔(指令数)
    support_edge_opt: bool = True  # 支持边界处理

    def get_kernel_name(self, mr: int, kr: int, nr: int) -> str:
        """
        获取内核函数名
        """
        return f"add_kernel_{self.platform}_{self.instruction_set}_{self.data_type.short_name}_{self.get_layout_buffer()}_mr{mr}_nr{nr}_kr{kr}"

    def get_main_kernel_name(self) -> str:
        """获取内核函数名"""
        return self.get_kernel_name(self.kernel_shapes.mr, self.kernel_shapes.kr, self.kernel_shapes.nr)

    def get_edge_kernel_name(self, edge_shape: KernelShape) -> str:
        """获取边界内核函数名"""
        return self.get_kernel_name(edge_shape.mr, edge_shape.kr, edge_shape.nr)

    def get_layout_buffer(self) -> str:
        """获取矩阵布局描述字符串"""
        def get_layout_chars(layout: MatrixLayout) -> str:
            """
            获取布局字符
            """
            if layout == MatrixLayout.ROW_MAJOR:
                return 'R'
            else:
                return 'C'
        return get_layout_chars(self.layout).lower()

    def get_dtype_c(self) -> str:
        """获取C语言数据类型"""
        return self.data_type.get_dtype_c()

    def get_dtype_lsl_position(self) -> int:
        """获取数据类型左移位数(用于向量寄存器索引计算)"""
        return self.data_type.get_dtype_lsl_position()

    def get_instruction_set_macro(self) -> str:
        """获取指令集宏定义"""
        macro_map = {
            'AVX2': '__AVX2__',
            'AVX512': '__AVX512F__',
            'NEON': '__ARM_NEON',
            'neon': '__ARM_NEON',
            'SVE': '__ARM_FEATURE_SVE',
            'SVE2': '__ARM_FEATURE_SVE2',
            'RVV': '__riscv_vector',
        }
        return macro_map.get(self.instruction_set, '')

    def get_platform_macro(self) -> str:
        """获取平台宏定义"""
        macro_map = {
            'AMD64': '__x86_64__',
            'X86': '__i386__',
            'ARMv8': '__aarch64__',
            'arm': '__aarch64__',
            'ARMv9': '__aarch64__',
            'RISC-V': '__riscv',
        }
        return macro_map.get(self.platform, '')

    @property
    def mr(self) -> int:
        """获取M维度分块大小"""
        return self.kernel_shapes.mr

    @property
    def nr(self) -> int:
        """获取N维度分块大小"""
        return self.kernel_shapes.nr

    @property
    def kr(self) -> int:
        """获取K维度分块大小"""
        return self.kernel_shapes.kr

    def get_mr_list(self) -> List[int]:
        """
        获取nr列表
        """
        mr_list = {self.mr}
        if self.support_edge_opt and self.kernel_shapes.edge_kernel_shapes:
            for shape in self.kernel_shapes.edge_kernel_shapes:
                mr_list.add(shape.mr)
        return sorted(list(mr_list), reverse=True)

    def get_nr_list(self, mr: Optional[int] = None) -> List[int]:
        """
        获取mr列表
        Args:
            mr: 指定mr值，若为None则获取所有mr值
        """
        nr_set = set()
        if mr is None or self.kernel_shapes.mr == mr:
            nr_set.add(self.kernel_shapes.nr)

        if self.support_edge_opt and self.kernel_shapes.edge_kernel_shapes:
            for shape in self.kernel_shapes.edge_kernel_shapes:
                if mr is None or shape.mr == mr:
                    nr_set.add(shape.nr)
        return sorted(list(nr_set), reverse=True)

    def get_kr_list(self, mr: Optional[int] = None, nr: Optional[int] = None) -> List[int]:
        """
        获取kr列表
        """
        kr_set = set()
        if (mr is None or self.kernel_shapes.mr == mr) and (nr is None or self.kernel_shapes.nr == nr):
            kr_set.add(self.kernel_shapes.kr)
        if self.support_edge_opt and self.kernel_shapes.edge_kernel_shapes:
            for shape in self.kernel_shapes.edge_kernel_shapes:
                if (mr is None or shape.mr == mr) and (nr is None or shape.nr == nr):
                    kr_set.add(shape.kr)
        return sorted(list(kr_set), reverse=True)

    def get_kernel_info(self) -> Dict[str, Any]:
        """
        获取内核信息字典
        """
        return {
            "platform": self.platform,
            "instruction_set": self.instruction_set,
            "data_type": self.data_type.short_name,
            "mr": self.mr,
            "nr": self.nr,
            "kr": self.kr,
            "layout": self.layout.value,
            "c_prefetch": self.c_prefetch.value,
            "partial_sum_prefetch": self.partial_sum_prefetch.value,
            "vec_reg_reuse_interval": self.vec_reg_reuse_interval,
            "support_edge_opt": self.support_edge_opt,
            "edge_shapes": self.kernel_shapes.get_edge_shapes_str(),
            "edge_shape_num": len(self.kernel_shapes.edge_kernel_shapes) if self.kernel_shapes.edge_kernel_shapes else 0,
            "function_name": self.get_main_kernel_name(),
            "platform_macro": self.get_platform_macro(),
            "instruction_set_macro": self.get_instruction_set_macro(),
        }
