# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/5 10:46
# Contact       : hucheng.lew@qq.com
# File          : ml_gemm_op.py
# IDE           : PyCharm
# Description   : 自定义的机器学习优化的矩阵乘法库

import tvm
from typing import List
from tvm import te, tir, s_tir, relax

def define_and_schedule_gemm(M: relax.Var , N: relax.Var, K: relax.Var, dtype: str = "float32") -> s_tir.Schedule:
    """
    定义矩阵乘法并进行调度
    """
    # TE 定义 (标准 GEMM)
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    # 使用 te.extern 直接将整个矩阵运算 offload 到自定义的 packed_func

    def fcompute(inputs: List[tir.Buffer], outputs: List[tir.Buffer]) -> tir.PrimExpr:
        # inputs[0] 是 A 的 Buffer, inputs[1] 是 B 的 Buffer, outputs[0] 是 C
        return tir.call_packed(
            "tvm.contrib.custom.te_simple_matmul",
            inputs[0], inputs[1], outputs[0],
            tir.const(1.0, dtype), tir.const(0.0, dtype)
        )

    C = te.extern(
        shape=(M, N),
        inputs=[A, B],
        fcompute=fcompute,
        name="C",
        dtype=dtype
    )

    func = te.create_prim_func([A, B, C])
    return s_tir.Schedule(func)


def define_and_schedule_gemm_with_bias(M: relax.Var , N: relax.Var, K: relax.Var, dtype: str = "float32") -> s_tir.Schedule:
    """
    生成 GEMM + Bias 的 TIR Schedule。
    GEMM 部分调用您的底层算子，Bias 加法部分保留 TVM 的循环，以便 TVM 进行并行化/向量化优化。
    """
    A = te.placeholder((M, K), name="A", dtype=dtype)
    B = te.placeholder((K, N), name="B", dtype=dtype)
    Bias = te.placeholder((N,), name="Bias", dtype=dtype)

    # 1. 第一步：调用你的底层大矩阵乘法
    def fcompute(inputs: List[tir.Buffer], outputs: List[tir.Buffer]) -> tir.PrimExpr:
        return tir.call_packed(
            "tvm.contrib.custom.te_simple_matmul",
            inputs[0], inputs[1], outputs[0],
            tir.const(1.0, dtype), tir.const(0.0, dtype)
        )

    Matmul_out = te.extern(
        shape=(M, N),
        inputs=[A, B],
        fcompute=fcompute,
        name="Matmul_out",
        dtype=dtype
    )

    # 2. 第二步：执行 Bias 加法
    # 这里使用 TVM 原生的 compute，TVM 会自动生成循环代码，并融合后续可能的激活函数等操作
    C = te.compute(
        (M, N),
        lambda i, j: Matmul_out[i, j] + Bias[j],
        name="C"
    )

    func = te.create_prim_func([A, B, Bias, C])
    sch = s_tir.Schedule(func)

    # (可选) 让 TVM 自动并行化 Bias 加法的外层循环，压榨 CPU 多核性能
    try:
        block_c = sch.get_block("C")
        i, j = sch.get_loops(block_c)
        sch.parallel(i)
    except Exception:
        pass

    return sch
