# -*- coding: utf-8 -*-
"""
统一入口点 - GEMM内核代码生成器

用法:
    python run_gemm_kernel_codegen.py --platform x86 --instruction-set avx2 \\
        --data-type f32 --mr 32 --nr 16 --kr 4 --output-dir output
"""

import sys
import argparse
from importlib import import_module
from typing import Optional

from gemm_codegen.kernel.gemm.base import (
    KernelParameters,
    PackingStrategy,
    KernelShapes
)

from gemm_codegen.common import (
    parse_precision, ElementPrecision,
    parse_layout, MatrixLayout,
    parse_prefetch, PrefetchCacheLevel,
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

    kernel_codegen_module_path = f"gemm_codegen.kernel.gemm.{platform_lower}"

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


def parse_packing(packing_str: str) -> PackingStrategy:
    """解析打包策略字符串"""
    if packing_str.lower() == 'none':
        return PackingStrategy.NO_PACKING
    elif packing_str.lower() == 'fine':
        return PackingStrategy.FINE_PACKING
    elif packing_str.lower() == 'coarse':
        return PackingStrategy.COARSE_PACKING
    else:
        raise ValueError(f"不支持的打包策略: {packing_str}")


def main():
    parser = argparse.ArgumentParser(
        description="GEMM内核汇编代码生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 生成x86 AVX2内核
  python run_gemm_kernel_codegen.py --platform x86 --instruction-set avx2 \\
      --data-type f32 --mr 32 --nr 16 --kr 4

  # 生成ARM NEON内核，启用预取
  python run_gemm_kernel_codegen.py --platform arm --instruction-set neon \\
      --data-type f32 --mr 4 --nr 64 --kr 8 \\
      --a-prefetch --b-prefetch

  # 生成x86 AVX-512内核，禁用边界处理
  python run_gemm_kernel_codegen.py --platform x86 --instruction-set avx512 \\
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
                       help="K维度分块大小")

    # 可选参数 - 矩阵布局和转置
    parser.add_argument("--c-layout", default="rowmajor",
                       help="C矩阵布局 (rowmajor/colmajor, 默认: rowmajor)")
    parser.add_argument("--a-layout", default="rowmajor",
                       help="A矩阵布局 (rowmajor/colmajor, 默认: rowmajor)")
    parser.add_argument("--b-layout", default="rowmajor",
                       help="B矩阵布局 (rowmajor/colmajor, 默认: rowmajor)")

    # 可选参数 - 访存优化
    parser.add_argument("--a-packing", default="none",
                       choices=['none', 'fine', 'coarse'],
                       help="矩阵A数据打包策略 (默认: none)")
    parser.add_argument("--b-packing", default="none",
                       choices=['none', 'fine', 'coarse'],
                       help="矩阵B数据打包策略 (默认: none)")
    parser.add_argument("--c-prefetch", default="none",
                        choices=["none", 'L1D', 'L2', 'L3'],
                        help="预取C矩阵到哪个缓存级别 (默认: none)")
    parser.add_argument("--a-prefetch", default="none",
                        choices=["none", 'L1D', 'L2', 'L3'],
                        help="预取A矩阵到哪个缓存级别 (默认: none)")
    parser.add_argument("--b-prefetch", default="none",
                        choices=["none", 'L1D', 'L2', 'L3'],
                        help="预取B矩阵到哪个缓存级别 (默认: none)")

    # 可选参数 - 边界处理
    parser.add_argument("--edge-optimization", action="store_true",
                       dest="support_edge_opt",
                       help="边界优化")

    # 可选参数 - 寄存器优化
    parser.add_argument("--vec-reg-reuse-interval", type=int, default=3,
                       help="向量寄存器重用间隔(指令数, 默认: 3)")
    parser.add_argument("--no-reg-spill", action="store_false",
                       dest="support_reg_spill",
                       help="禁用通用寄存器溢出")

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
        c_layout = parse_layout(args.c_layout)
        a_layout = parse_layout(args.a_layout)
        b_layout = parse_layout(args.b_layout)
        a_packing = parse_packing(args.a_packing)
        b_packing = parse_packing(args.b_packing)
        c_prefetch = parse_prefetch(args.c_prefetch)
        b_prefetch = parse_prefetch(args.b_prefetch)
        a_prefetch = parse_prefetch(args.a_prefetch)
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
        c_layout=c_layout,
        a_layout=a_layout,
        b_layout=b_layout,
        a_packing=a_packing,
        b_packing=b_packing,
        c_prefetch=c_prefetch,
        a_prefetch=a_prefetch,
        b_prefetch=b_prefetch,
        vec_reg_reuse_interval=args.vec_reg_reuse_interval,
        support_reg_spill=args.support_reg_spill,
        support_edge_opt=args.support_edge_opt,
    )

    if args.verbose:
        print(f"\n{'='*60}")
        print(f"GEMM内核代码生成器")
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

    # 分配寄存器
    try:
        register_allocation = generator.allocate_registers()

        # if args.show_register_allocation:
        #     print(register_allocation.get_utilization_report())
        #
        # if args.verbose:
        #     print(f"向量寄存器使用率: {register_allocation.vector_regs_used}/{register_allocation.total_vector_regs}")
        #     print(f"通用寄存器使用率: {register_allocation.general_regs_used}/{register_allocation.total_general_regs}")
        #     if register_allocation.spilled_regs:
        #         print(f"警告: {len(register_allocation.spilled_regs)} 个寄存器需要溢出: {register_allocation.spilled_regs}")
        #     print()
    except Exception as e:
        print(f"错误: 寄存器分配失败")
        print(f"详细信息: {e}")
        sys.exit(1)

    # 生成代码
    try:
        if args.verbose:
            print("正在生成汇编代码...")
        assembly_path = generator.save_assembly(args.output_dir)
        print(f"✓ 汇编代码: {assembly_path}")

        # if args.verbose:
        #     print("正在生成头文件...")
        # header_path = generator.save_header(args.output_dir)
        # print(f"✓ 头文件: {header_path}")
        #
        # if args.verbose:
        #     print("正在生成测试代码...")
        # test_path = generator.save_test(args.output_dir)
        # print(f"✓ 测试代码: {test_path}")
        #
        # print(f"\n生成成功!")
        #
        # # 提供编译建议
        # print(f"\n编译建议:")
        # if args.platform == 'x86':
        #     print(f"  汇编: as {os.path.basename(assembly_path)} -o kernel.o")
        #     if args.instruction_set == 'avx2':
        #         print(f"  测试: g++ -O3 -mavx2 {os.path.basename(test_path)} kernel.o -o test_kernel")
        #     elif args.instruction_set == 'avx512':
        #         print(f"  测试: g++ -O3 -mavx512f {os.path.basename(test_path)} kernel.o -o test_kernel")
        # elif args.platform == 'arm':
        #     print(f"  汇编: aarch64-linux-gnu-as {os.path.basename(assembly_path)} -o kernel.o")
        #     print(f"  测试: aarch64-linux-gnu-g++ -O3 {os.path.basename(test_path)} kernel.o -o test_kernel")

    except Exception as e:
        print(f"错误: 代码生成失败")
        print(f"详细信息: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
