# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/23 22:34
# Contact       : hucheng.lew@qq.com
# File          : run_add_kernel_codegen.py
# IDE           : PyCharm
# Description   : 运行加法内核代码生成器


import sys
import argparse
from importlib import import_module
from typing import Optional


from gemm_codegen.common import (
    parse_layout, parse_prefetch, parse_precision,
    MatrixLayout, PrefetchCacheLevel
)

from .base import (
    KernelParameters, KernelShapes
)


def get_kernel_generator_factory(platform: str) -> Optional[callable]:
    """根据平台和指令集获取生成器类

    Args:
        platform: 平台名称(x86, arm, riscv等)
        instruction_set: 指令集名称(avx2, avx512, neon, sve, sve2, sme, amx, amd_amx等)

    Returns:
        生成器类
    """
    # 构造模块路径
    platform_lower = platform.lower()

    kernel_codegen_module_path = f"gemm_codegen.kernel.add.{platform_lower}"

    try:
        module = import_module(kernel_codegen_module_path)

        kernel_generator_factory = getattr(module, "kernel_generator_factory")
        return kernel_generator_factory

    except ImportError as e:
        print(f"错误: 找不到生成器模块 {kernel_codegen_module_path}")
        print(f"详细信息: {e}")
        print(f"\n可用的平台和指令集组合:")
        print(f"  x86: avx2, avx512, amx, amd_amx")
        print(f"  arm: neon, sve, sve2, sme")
        print(f"  riscv: rvv")
        return None
    except AttributeError as e:
        print(f"错误: 模块 {kernel_codegen_module_path} 中找不到函数 kernel_generator_factory")
        print(f"详细信息: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="ADD内核汇编代码生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成x86 AVX2内核
  python run_add_kernel_codegen.py --platform x86 --instruction-set avx2 \\
      --data-type f32 --mr 32 --nr 16 --kr 4

  # 生成ARM NEON内核，启用预取
  python run_add_kernel_codegen.py --platform arm --instruction-set neon \\
      --data-type f32 --mr 4 --nr 64 --kr 8 \\
      --a-prefetch --b-prefetch

  # 生成x86 AVX-512内核，禁用边界处理
  python run_add_kernel_codegen.py --platform x86 --instruction-set avx512 \\
      --data-type f64 --mr 16 --nr 32 --kr 4 \\
      --no-edge-handling
        """
    )

    # 必需参数
    parser.add_argument("--platform", required=True,
                        choices=['x86', 'arm', 'riscv'],
                        help="目标平台架构")
    parser.add_argument("--instruction-set", required=True,
                        help="指令集扩展 (x86: avx2, avx512, amx, amd_amx | arm: neon, sve, sve2, sme | riscv: rvv)")
    parser.add_argument("--data-type", required=True,
                        choices=['f8', 'f16', 'f32', 'f64', 'i8', 'i16', 'i32', 'i64'],
                        help="数据类型")
    parser.add_argument("--mr", type=int, required=True,
                        help="M维度分块大小")
    parser.add_argument("--nr", type=int, required=True,
                        help="N维度分块大小")
    parser.add_argument("--kr", type=int, required=True,
                        help="一次最多累加部分和的数量")

    # 可选参数 - 矩阵布局和转置
    parser.add_argument("--layout", default="rowmajor",
                        help="C矩阵布局 (rowmajor/colmajor, 默认: rowmajor)")

    # 可选参数 - 访存优化
    parser.add_argument("--c-prefetch", default="none",
                        choices=["none", 'L1D', 'L2', 'L3'],
                        help="预取C矩阵到哪个缓存级别 (默认: none)")
    parser.add_argument("--partial-sum-prefetch", default="none",
                        choices=["none", 'L1D', 'L2', 'L3'],
                        help="预取部分和矩阵到哪个缓存级别 (默认: none)")

    # 可选参数 - 边界处理
    parser.add_argument("--edge-optimization", action="store_true",
                        dest="support_edge_opt",
                        help="边界优化")

    # 可选参数 - 寄存器优化
    parser.add_argument("--vec-reg-reuse-interval", type=int, default=3,
                        help="向量寄存器重用间隔(指令数, 默认: 3)")

    # 输出选项
    parser.add_argument("--output-dir", default="output",
                        help="输出目录 (默认: output)")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")
    parser.add_argument("--show-register-allocation", action="store_true",
                        help="显示寄存器分配报告")

    args = parser.parse_args()

    # 解析参数
    try:
        layout = parse_layout(args.layout)
        c_prefetch = parse_prefetch(args.c_prefetch)
        partial_sum_prefetch = parse_prefetch(args.partial_sum_prefetch)
        data_type = parse_precision(args.data_type)
    except ValueError as e:
        print(f"参数错误: {e}")
        sys.exit(1)

    # 创建内核参数
    params = KernelParameters(
        platform=args.platform,
        instruction_set=args.instruction_set.lower(),
        data_type=data_type,
        kernel_shapes = KernelShapes(
            mr=args.mr,
            nr=args.nr,
            kr=args.kr,
        ),
        layout=layout,
        c_prefetch=c_prefetch,
        partial_sum_prefetch=partial_sum_prefetch,
        vec_reg_reuse_interval=args.vec_reg_reuse_interval,
        support_edge_opt=args.support_edge_opt,
    )

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"ADD内核代码生成器")
        print(f"{'='*60}")
        print(f"平台: {params.platform}")
        print(f"指令集: {params.instruction_set}")
        print(f"数据类型: {params.data_type}")
        print(f"分块大小: mr={params.mr}, nr={params.nr}, kr={params.kr}")
        print(f"内核函数名: {params.get_main_kernel_name()}")
        print(f"{'='*60}\n")

    # 获取生成器类
    kernel_generator_factory = get_kernel_generator_factory(args.platform)
    if not kernel_generator_factory:
        sys.exit(1)

    # 创建生成器实例
    try:
        generator = kernel_generator_factory(params)
    except Exception as e:
        print(f"错误: 创建生成器失败")
        print(f"详细信息: {e}")
        sys.exit(1)

    # 生成代码
    try:
        if args.verbose:
            print("正在生成汇编代码...")
        assembly_path = generator.save_assembly(args.output_dir)
        print(f"✓ 汇编代码: {assembly_path}")

    except Exception as e:
        print(f"错误: 代码生成失败")
        print(f"详细信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
