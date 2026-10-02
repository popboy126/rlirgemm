# -*- coding: utf-8 -*-
"""
加法内核生成器基类

提供通用的寄存器分配算法、模板渲染机制和代码生成流程控制。
所有平台特定的生成器都应继承此基类并实现抽象方法。
"""

import os
import jinja2
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Generator
from codecs import open
from .register_allocation import RegisterAllocation
from .kernel import KernelParameters
from string import Template
from gemm_codegen.common import (
    ResultScaleType
)



class BaseAddKernelGenerator(ABC):
    """加法内核生成器基类

    提供通用的代码生成流程和工具方法，具体平台的实现通过继承此类完成。
    """

    def __init__(self, params: KernelParameters):
        """初始化生成器

        Args:
            params: 内核参数配置
        """
        self.params = params

        # 设置模板环境
        self.template_base = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '..', '..', '..',  'templates', 'add'
        ))

        self.template_dir = os.path.join(
            self.template_base,
            params.platform.lower(),
            params.instruction_set.lower()
        )

        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([
                self.template_base,
                self.template_dir
            ]),
            trim_blocks=True,
            lstrip_blocks=True
        )

        # 寄存器分配结果
        self.register_allocation: Optional[RegisterAllocation] = None

    def get_result_scale_type(self) -> ResultScaleType:
        """获取结果缩放类型"""
        return self.__result_scale_type

    def __set_result_scale_type(self, scale_type: ResultScaleType):
        """设置结果缩放类型"""
        self.__result_scale_type = scale_type

    def _get_element_size(self) -> int:
        """获取数据类型的大小(bits)"""
        return self.params.data_type.get_bit_size()


    @abstractmethod
    def get_vector_register_count(self) -> int:
        """获取平台可用的向量寄存器总数"""
        pass

    @abstractmethod
    def get_general_register_count(self) -> int:
        """获取平台可用的通用寄存器总数"""
        pass

    @abstractmethod
    def get_vector_register_names(self) -> List[str]:
        """获取向量寄存器名称列表"""
        pass

    @abstractmethod
    def get_general_register_names(self) -> List[str]:
        """获取通用寄存器名称列表"""
        pass

    def general_register_allocator(self) -> Generator[str, Any, None]:
        """通用寄存器分配器"""
        for i in self.get_general_register_names():
            yield i

    def vector_register_allocator(self) -> Generator[str, Any, None]:
        """向量寄存器分配器"""
        for i in self.get_vector_register_names():
            yield i

    def get_elements_per_vector_reg(self) -> int:
        """获取每个向量寄存器可容纳的元素数量"""
        vector_width = self._get_vector_width()
        element_size = self._get_element_size()
        return vector_width // element_size

    @abstractmethod
    def allocate_registers(self) -> RegisterAllocation:
        """分配寄存器
        按照优化策略分配向量寄存器和通用寄存器
        """
        pass

    @abstractmethod
    def _get_vector_width(self) -> int:
        """获取向量宽度(bits)"""
        pass

    @abstractmethod
    def generate_function_prologue(self) -> List[str]:
        """生成函数序言"""
        pass

    @abstractmethod
    def generate_function_epilogue(self) -> str:
        """生成函数尾声"""
        pass

    @abstractmethod
    def generate_kernel_shapes(self) -> List[Tuple[int, int, int]]:
        """生成边界内核形状处理代码"""
        pass

    def get_label_suffix(self) -> str:
        """获取标签后缀"""
        return f"with_{self.get_result_scale_type().value}".upper()


    def get_kernel_macro_fmt(self) -> Template:
        """获取内核计算体宏模板"""
        return Template("KERNEL_MR${mr}_NR${nr}_KR${kr}" + "_" + self.get_label_suffix())

    def generate_main_loop(self) -> Dict[str, List[str] | Any]:
        """生成主计算循环"""
        self.__set_result_scale_type(ResultScaleType.ALPHA1_BETA0)
        main_loop_with_alpha1_beta0 = self.get_main_loop_with_alpha1_beta0()

        self.__set_result_scale_type(ResultScaleType.ALPHA1_BETA1)
        main_loop_with_alpha1_beta1 = self.get_main_loop_with_alpha1_beta1()

        self.__set_result_scale_type(ResultScaleType.ALPHA1_BETA)
        main_loop_with_alpha1_beta = self.get_main_loop_with_alpha1_beta()

        self.__set_result_scale_type(ResultScaleType.ALPHA_BETA0)
        main_loop_with_alpha_beta0 = self.get_main_loop_with_alpha_beta0()

        self.__set_result_scale_type(ResultScaleType.ALPHA_BETA1)
        main_loop_with_alpha_beta1 = self.get_main_loop_with_alpha_beta1()

        self.__set_result_scale_type(ResultScaleType.ALPHA_BETA)
        main_loop_with_alpha_beta = self.get_main_loop_with_alpha_beta()
        return {
            self.get_main_loop_with_alpha1_beta0_name(): main_loop_with_alpha1_beta0,
            self.get_main_loop_with_alpha1_beta1_name(): main_loop_with_alpha1_beta1,
            self.get_main_loop_with_alpha1_beta_name(): main_loop_with_alpha1_beta,
            self.get_main_loop_with_alpha_beta0_name(): main_loop_with_alpha_beta0,
            self.get_main_loop_with_alpha_beta1_name(): main_loop_with_alpha_beta1,
            self.get_main_loop_with_alpha_beta_name(): main_loop_with_alpha_beta,
        }

    def get_main_loop_with_alpha1_beta0_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=1, beta=0"""
        return "main_loop_with_alpha1_beta0".upper()

    @abstractmethod
    def get_main_loop_with_alpha1_beta0(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=0"""
        pass

    def get_main_loop_with_alpha1_beta1_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=1, beta=1"""
        return "main_loop_with_alpha1_beta1".upper()

    @abstractmethod
    def get_main_loop_with_alpha1_beta1(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=1"""
        pass

    def get_main_loop_with_alpha1_beta_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=1, beta=*"""
        return "main_loop_with_alpha1_beta".upper()

    @abstractmethod
    def get_main_loop_with_alpha1_beta(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=1, beta=*"""
        pass

    def get_main_loop_with_alpha_beta0_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=*, beta=0"""
        return "main_loop_with_alpha_beta0".upper()

    @abstractmethod
    def get_main_loop_with_alpha_beta0(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=0"""
        pass

    def get_main_loop_with_alpha_beta1_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=*, beta=1"""
        return "main_loop_with_alpha_beta1".upper()

    @abstractmethod
    def get_main_loop_with_alpha_beta1(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=1"""
        pass

    def get_main_loop_with_alpha_beta_name(self) -> str:
        """获取主计算循环函数名，假设 alpha=*, beta=*"""
        return "main_loop_with_alpha_beta".upper()

    @abstractmethod
    def get_main_loop_with_alpha_beta(self) -> Dict[str, List[str] | Any]:
        """获取主计算循环，假设 alpha=*, beta=*"""
        pass


    @abstractmethod
    def generate_main_loop_scheduler(self) -> Dict[str, Dict[str, List[str] | Any]]:
        """
        生成主计算循环调度器
        """
        pass

    def generate_assembly(self) -> str:
        """生成完整的汇编代码"""
        # 确保已生成所有边界内核形状
        if self.params.support_edge_opt and self.params.kernel_shapes.edge_kernel_shapes is None:
            self.params.kernel_shapes.set_edge_shapes(self.generate_kernel_shapes())
        # 确保已分配寄存器
        if self.register_allocation is None:
            self.register_allocation = self.allocate_registers()

        # 准备模板上下文
        context = self._prepare_assembly_context()

        # 渲染模板
        template = self.env.get_template('kernel/kernel.S.jinja')
        return template.render(context)

    def generate_header(self) -> str:
        """生成C/C++头文件"""
        context = self._prepare_header_context()
        template = jinja2.Template("""#ifndef {{ kernel_name|upper }}_H
#define {{ kernel_name|upper }}_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void {{ kernel_name }}(
    {{ dtype_c }}* C,
    int64_t ldc,
    {{ dtype_c }}** partial_sum_ptr,
    int64_t ldp,
    int64_t m,
    int64_t n,
    int64_t k,
    {{ dtype_c }} alpha,
    {{ dtype_c }} beta,
    int64_t c_prefetch_stride,
    int64_t partial_sum_prefetch_stride
);

#ifdef __cplusplus
}
#endif

#endif // {{ kernel_name|upper }}_H

        """
        )
        return template.render(context)

    def _prepare_assembly_context(self) -> Dict:
        """准备汇编代码模板上下文"""
        return {
            'KernelInfo': self.params.get_kernel_info(),
            'RegisterAllocation': self.register_allocation.get_register_allocation_report(),
            'function_name': self.params.get_main_kernel_name(),
            'platform': self.params.platform,
            'instruction_set': self.params.instruction_set,
            'data_type': self.params.data_type,
            'mr': self.params.mr,
            'nr': self.params.nr,
            'kr': self.params.kr,
            'c_prefetch': self.params.c_prefetch,
            'partial_sum_prefetch': self.params.partial_sum_prefetch,
            'register_allocation': self.register_allocation,
            'c_header': self.generate_header(),
            'prologue': self.generate_function_prologue(),
            'epilogue': self.generate_function_epilogue(),
            'main_loop': self.generate_main_loop(),
            'main_loop_scheduler': self.generate_main_loop_scheduler(),
        }

    def _prepare_header_context(self) -> Dict:
        """准备头文件模板上下文"""
        return {
            'kernel_name': self.params.get_main_kernel_name(),
            'dtype_c': self.params.get_dtype_c(),
        }

    @abstractmethod
    def _get_instruction_set_macro(self) -> str:
        """获取指令集宏定义"""
        pass

    def save_assembly(self, output_dir: str) -> str:
        """保存汇编代码到文件

        Args:
            output_dir: 输出目录

        Returns:
            生成的文件路径
        """
        code = self.generate_assembly()

        # 创建输出目录
        kernel_dir = os.path.join(
            output_dir, 'kernels',
            self.params.platform.lower(),
            self.params.instruction_set.lower()
        )
        os.makedirs(kernel_dir, exist_ok=True)

        # 保存文件
        filename = f"{self.params.get_main_kernel_name()}.S"
        filepath = os.path.join(kernel_dir, filename)

        with open(filepath, 'w', encoding="utf-8") as f:
            f.write(code)

        return filepath
