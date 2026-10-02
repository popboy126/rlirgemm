# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/11 21:32
# Contact       : hucheng.lew@qq.com
# File          : kernel.py
# IDE           : PyCharm
# Description   : 跟内核定义有关的数据结果

from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from gemm_codegen.common import (
    ElementPrecision, MatrixLayout, PrefetchCacheLevel
)

class PackingStrategy(Enum):
    """数据打包策略"""
    NO_PACKING = "NoPacking"
    FINE_PACKING = "FinePacking"
    COARSE_PACKING = "CoarsePacking"

class KernelAlgorithm(Enum):
    """内核计算算法"""
    # 外积算法
    OUTER_PRODUCT = "OuterProduct"
    # 内积算法
    INNER_PRODUCT = "InnerProduct"



@dataclass
class KernelShape:
    """内核形状"""
    mr: int
    kr: int
    nr: int

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
        self._edge_shapes = [KernelShape(mr, kr, nr) for (mr, kr, nr) in edge_shapes]

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
    c_layout: MatrixLayout = MatrixLayout.ROW_MAJOR
    a_layout: MatrixLayout = MatrixLayout.ROW_MAJOR
    b_layout: MatrixLayout = MatrixLayout.ROW_MAJOR

    # 访存优化
    a_packing: PackingStrategy = PackingStrategy.NO_PACKING
    b_packing: PackingStrategy = PackingStrategy.NO_PACKING
    c_prefetch: PrefetchCacheLevel = PrefetchCacheLevel.NONE
    a_prefetch: PrefetchCacheLevel = PrefetchCacheLevel.NONE
    b_prefetch: PrefetchCacheLevel = PrefetchCacheLevel.NONE

    # 寄存器优化
    vec_reg_reuse_interval: int = 3  # 寄存器重用间隔(指令数)
    support_reg_spill: bool = True  # 支持通用寄存器溢出
    support_edge_opt: bool = True  # 支持边界处理

    def get_kernel_name(self, mr: int, kr: int, nr: int) -> str:
        """
        获取内核函数名
        """
        return f"mm_kernel_{self.platform}_{self.instruction_set}_{self.data_type.short_name}_{self.get_layout_buffer()}_{self.get_packing_buffer()}_mr{mr}_nr{nr}_kr{kr}"

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

        a_layout_str = get_layout_chars(self.a_layout)
        b_layout_str = get_layout_chars(self.b_layout)
        c_layout_str = get_layout_chars(self.c_layout)
        return f"c{c_layout_str}_a{a_layout_str}_b{b_layout_str}".lower()

    def get_packing_buffer(self) -> str:
        """获取矩阵打包描述字符串"""
        def get_packing_chars(packing: PackingStrategy) -> str:
            """
            获取打包字符
            """
            if packing == PackingStrategy.NO_PACKING:
                return 'NP'
            elif packing == PackingStrategy.FINE_PACKING:
                return 'FP'
            else:
                return 'CP'

        a_packing_str = get_packing_chars(self.a_packing)
        b_packing_str = get_packing_chars(self.b_packing)
        return f"a{a_packing_str}_b{b_packing_str}".lower()

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

    def get_algorithm_type(self) -> KernelAlgorithm:
        """
        获取内核的算法
        """
        key = (self.c_layout, self.a_layout, self.b_layout)
        algorithm_mapping = {
            (MatrixLayout.ROW_MAJOR, MatrixLayout.ROW_MAJOR, MatrixLayout.ROW_MAJOR): KernelAlgorithm.OUTER_PRODUCT,
            (MatrixLayout.ROW_MAJOR, MatrixLayout.ROW_MAJOR, MatrixLayout.COL_MAJOR): KernelAlgorithm.INNER_PRODUCT,
            (MatrixLayout.ROW_MAJOR, MatrixLayout.COL_MAJOR, MatrixLayout.ROW_MAJOR): KernelAlgorithm.OUTER_PRODUCT,
            (MatrixLayout.ROW_MAJOR, MatrixLayout.COL_MAJOR, MatrixLayout.COL_MAJOR): KernelAlgorithm.OUTER_PRODUCT,
            (MatrixLayout.COL_MAJOR, MatrixLayout.ROW_MAJOR, MatrixLayout.ROW_MAJOR): KernelAlgorithm.INNER_PRODUCT,
            (MatrixLayout.COL_MAJOR, MatrixLayout.ROW_MAJOR, MatrixLayout.COL_MAJOR): KernelAlgorithm.OUTER_PRODUCT,
            (MatrixLayout.COL_MAJOR, MatrixLayout.COL_MAJOR, MatrixLayout.ROW_MAJOR): KernelAlgorithm.INNER_PRODUCT,
            (MatrixLayout.COL_MAJOR, MatrixLayout.COL_MAJOR, MatrixLayout.COL_MAJOR): KernelAlgorithm.OUTER_PRODUCT,
        }
        return algorithm_mapping[key]

    def is_matrix_contiguous(self, matrix: str) -> bool:
        """判断矩阵访问是否连续

        Args:
            matrix: 'A' or 'B' or 'C'

        Returns:
            矩阵是否能连续访问
        """
        kernel_alg = self.get_algorithm_type()
        if matrix == 'A':
            if kernel_alg == KernelAlgorithm.OUTER_PRODUCT:
                # A矩阵按行访问(遍历K)
                if self.a_layout == MatrixLayout.ROW_MAJOR:
                    return False  # 行优先，按行访问，连续
                return True
            else:
                # A矩阵按列访问(遍历K)
                if self.a_layout == MatrixLayout.ROW_MAJOR:
                    return True
                return False
        elif matrix == 'B':
            if kernel_alg == KernelAlgorithm.OUTER_PRODUCT:
                # B矩阵按列访问(遍历K)
                if self.b_layout == MatrixLayout.ROW_MAJOR:
                    return True
                return False
            else:
                # B矩阵按行访问(遍历K)
                if self.b_layout == MatrixLayout.ROW_MAJOR:
                    return False
                return True
        else:  # 矩阵C默认访问是不连续的
            return False

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

    def get_nr_list(self) -> List[int]:
        """
        获取nr列表
        """
        nr_list = {self.nr}
        if self.support_edge_opt and self.kernel_shapes.edge_kernel_shapes:
            for shape in self.kernel_shapes.edge_kernel_shapes:
                nr_list.add(shape.nr)
        return sorted(list(nr_list), reverse=True)

    def get_mr_list(self, nr: Optional[int] = None) -> List[int]:
        """
        获取mr列表
        Args:
            nr: 指定nr值，若为None则获取所有mr值
        """
        mr_set = set()
        if nr is None or self.kernel_shapes.nr == nr:
            mr_set.add(self.kernel_shapes.mr)
        if self.support_edge_opt and self.kernel_shapes.edge_kernel_shapes:
            for shape in self.kernel_shapes.edge_kernel_shapes:
                if nr is None or shape.nr == nr:
                    mr_set.add(shape.mr)
        return sorted(list(mr_set), reverse=True)

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
            "c_layout": self.c_layout.value,
            "a_layout": self.a_layout.value,
            "b_layout": self.b_layout.value,
            "a_packing": self.a_packing.value,
            "b_packing": self.b_packing.value,
            "c_prefetch": self.c_prefetch.value,
            "a_prefetch": self.a_prefetch.value,
            "b_prefetch": self.b_prefetch.value,
            "vec_reg_reuse_interval": self.vec_reg_reuse_interval,
            "support_reg_spill": self.support_reg_spill,
            "support_edge_opt": self.support_edge_opt,
            "edge_shapes": self.kernel_shapes.get_edge_shapes_str(),
            "edge_shape_num": len(self.kernel_shapes.edge_kernel_shapes) if self.kernel_shapes.edge_kernel_shapes else 0,
            "function_name": self.get_main_kernel_name(),
            "platform_macro": self.get_platform_macro(),
            "instruction_set_macro": self.get_instruction_set_macro(),
        }
