# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/2 10:35
# Contact       : hucheng.lew@qq.com
# File          : compile_qwen3_4b_model.py
# IDE           : PyCharm
# Description   : 编译qwen3 4B模型

import json
import logging
import os
import sys
import time

# import mlc_llm.cli.compile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import tvm
from ml_gemm_op import define_and_schedule_gemm, define_and_schedule_gemm_with_bias
from mlc_llm import MLCEngine
from mlc_llm.interface.compile import (
    CompileArgs,
    ModelConfigOverride,
    OptimizationFlags,
    compile,
)
from mlc_llm.interface.convert_weight import convert_weight
from mlc_llm.interface.gen_config import CONV_TEMPLATES, gen_config
from mlc_llm.model import MODELS
from mlc_llm.quantization import QUANTIZATION
from mlc_llm.support.auto_target import _add_system_lib_prefix, _build_default
from mlc_llm.support.auto_weight import detect_weight
from tvm import IRModule, relax, te, tir
from tvm.contrib import cblas
from tvm.runtime import Device

logger = logging.getLogger(__name__)


def init_logs(log_dir: str):
    """
    初始化日志
    :param log_dir:
    :return:
    """
    # 以下是初始化log
    # 创建logger对象
    logger.setLevel(logging.DEBUG)
    # 创建FileHandler对象
    fh = logging.FileHandler(
        os.path.join(log_dir, "model_compile.log"), encoding="utf-8"
    )
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # 创建Formatter对象
    formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s:%(lineno)d]  %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 将FileHandler对象添加到Logger对象中
    logger.addHandler(fh)
    logger.addHandler(ch)


def convert_qwen3_weights(
    model_type: str,
    weight_format: str,
    model_path: str,
    quantization: str,
    target: Device,
    output_dir: str,
) -> None:
    """
    转换千问3的权重
    """
    config_path = os.path.join(model_path, "config.json")
    config_path = Path(config_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    # 第一步: 转换权重和bias
    source, source_format = detect_weight(model_path, config_path, weight_format)
    convert_weight(
        config=config_path,
        quantization=QUANTIZATION[quantization],
        model=MODELS[model_type],
        device=target,
        source=source,
        source_format=source_format,
        output=output_dir,
    )
    logger.info(f"convert {model_type} weight successfully")


def gen_qwen3_configs(
    model_type: str,
    conv_template: str,
    model_path: str,
    quantization: str,
    output_dir: str,
) -> None:
    """
    获取千问3模式的配置
    """
    config_path = os.path.join(model_path, "config.json")
    config_path = Path(config_path)
    output_dir = Path(output_dir)
    model = MODELS[model_type]
    # conv_template = "chatml"

    gen_config(
        config=config_path,
        model=model,
        quantization=QUANTIZATION[quantization],
        conv_template=conv_template,
        context_window_size=None,
        sliding_window_size=None,
        prefill_chunk_size=None,
        attention_sink_size=None,
        tensor_parallel_shards=None,
        pipeline_parallel_stages=None,
        disaggregation=None,
        max_batch_size=128,
        output=output_dir,
    )
    logger.info(f"generate {model_type} config successfully")


def _custom_build_func(
    compile_type: str,
) -> Callable[[IRModule, CompileArgs, Any], None]:
    """
    自定义的构建函数（升级版：原生 Mutator 图融合）
    """

    def build(mod: IRModule, args: CompileArgs, pipeline=None):

        # =====================================================================
        # 第一部分：定义 Matmul+Bias 融合的 Pass (在 Legalize 之前运行)
        # =====================================================================
        @relax.expr_functor.mutator
        class FuseMatmulBiasMutator(relax.PyExprMutator):
            def __init__(self, mod: tvm.IRModule):
                super().__init__(mod)

            def visit_call_(self, call: relax.Call) -> relax.Expr:
                # 拦截所有 relax.add 操作
                if call.op == tvm.ir.Op.get("relax.add"):
                    arg0 = call.args[0]
                    arg1 = call.args[1]

                    # 追溯变量的原始绑定表达式（这一步直接代替了脆弱的全局列表）
                    bound_expr0 = (
                        self.lookup_binding(arg0)
                        if isinstance(arg0, relax.Var)
                        else arg0
                    )
                    bound_expr1 = (
                        self.lookup_binding(arg1)
                        if isinstance(arg1, relax.Var)
                        else arg1
                    )

                    matmul_expr = None
                    bias_expr = None

                    # 兼容 add 的交换律：检查 arg0 或 arg1 哪个是 matmul
                    if isinstance(
                        bound_expr0, relax.Call
                    ) and bound_expr0.op == tvm.ir.Op.get("relax.matmul"):
                        matmul_expr = bound_expr0
                        bias_expr = arg1
                    elif isinstance(
                        bound_expr1, relax.Call
                    ) and bound_expr1.op == tvm.ir.Op.get("relax.matmul"):
                        matmul_expr = bound_expr1
                        bias_expr = arg0

                    # 如果找到了完美的 add(matmul, bias) 模式
                    if matmul_expr is not None:
                        # 递归处理输入，确保上游也变异完成
                        A = self.visit_expr(matmul_expr.args[0])
                        B = self.visit_expr(matmul_expr.args[1])
                        Bias = self.visit_expr(bias_expr)

                        bias_shape = list(Bias.struct_info.shape)

                        # 确认 Bias 是一维的再融合
                        if len(bias_shape) == 1:
                            lhs_shape = list(A.struct_info.shape)
                            rhs_shape = list(B.struct_info.shape)
                            out_shape_nd = list(call.struct_info.shape)

                            K = lhs_shape[-1]
                            N = rhs_shape[-1]

                            analyzer = tvm.arith.Analyzer()
                            M = tvm.tir.const(1, "int64")
                            for s in lhs_shape[:-1]:
                                M = M * s
                            M = analyzer.simplify(M)

                            def te_custom_gemm_bias(A_te, B_te, Bias_te):
                                def fcompute(inputs, outputs):
                                    return tir.Evaluate(
                                        tir.call_packed(
                                            "tvm.contrib.custom.te_simple_matmul",
                                            inputs[0],
                                            inputs[1],
                                            outputs[0],
                                            tir.const(1.0, A_te.dtype),
                                            tir.const(0.0, A_te.dtype),
                                        )
                                    )

                                Matmul_out = te.extern(
                                    shape=(A_te.shape[0], B_te.shape[1]),
                                    inputs=[A_te, B_te],
                                    fcompute=fcompute,
                                    name="Matmul_out",
                                    dtype=A_te.dtype,
                                )
                                return te.compute(
                                    (A_te.shape[0], B_te.shape[1]),
                                    lambda i, j: Matmul_out[i, j] + Bias_te[j],
                                    name="C",
                                )

                            # 使用 Mutator 内置的 self.builder_ 构造并插入新节点
                            lhs_2d = self.builder_.emit(relax.op.reshape(A, [M, K]))

                            if len(rhs_shape) > 2:
                                rhs_K = tvm.tir.const(1, "int64")
                                for s in rhs_shape[:-1]:
                                    rhs_K = rhs_K * s
                                rhs_K = analyzer.simplify(rhs_K)
                                rhs_2d = self.builder_.emit(
                                    relax.op.reshape(B, [rhs_K, N])
                                )
                            else:
                                rhs_2d = B

                            out_2d_call = self.builder_.call_te(
                                te_custom_gemm_bias, lhs_2d, rhs_2d, Bias
                            )
                            out_2d_var = self.builder_.emit(out_2d_call)
                            final_reshape = relax.op.reshape(out_2d_var, out_shape_nd)

                            # 返回归一化后的新节点，自动替换掉原有的 add 节点
                            return self.builder_.normalize(final_reshape)

                # 对于非目标节点，执行默认的变异逻辑
                return super().visit_call_(call)

        @tvm.transform.module_pass(opt_level=0, name="FuseMatmulBiasPass")
        def fuse_matmul_bias_pass(
            mod: tvm.IRModule, ctx: tvm.transform.PassContext
        ) -> tvm.IRModule:
            mutator = FuseMatmulBiasMutator(mod)
            for gv, func in mod.functions.items():
                if isinstance(func, relax.Function):
                    updated_func = mutator.visit_expr(func)
                    mutator.builder_.update_func(gv, updated_func)
            return mutator.builder_.get()

        # =====================================================================
        # 第二部分：纯粹的 Matmul Legalizer（只处理未被融合的单独 matmul）
        # =====================================================================
        def custom_matmul_legalize(bb: relax.BlockBuilder, call: relax.Call):
            lhs, rhs = call.args[0], call.args[1]
            lhs_shape, rhs_shape = (
                list(lhs.struct_info.shape),
                list(rhs.struct_info.shape),
            )
            out_shape_nd = list(call.struct_info.shape)

            K, N = lhs_shape[-1], rhs_shape[-1]
            analyzer = tvm.arith.Analyzer()
            M = tvm.tir.const(1, "int64")
            for s in lhs_shape[:-1]:
                M = M * s
            M = analyzer.simplify(M)

            def te_custom_gemm(A, B):
                def fcompute(inputs, outputs):
                    return tir.Evaluate(
                        tir.call_packed(
                            "tvm.contrib.custom.te_simple_matmul",
                            inputs[0],
                            inputs[1],
                            outputs[0],
                            tir.const(1.0, A.dtype),
                            tir.const(0.0, A.dtype),
                        )
                    )

                return te.extern(
                    shape=(A.shape[0], B.shape[1]),
                    inputs=[A, B],
                    fcompute=fcompute,
                    name="C",
                    dtype=A.dtype,
                )

            lhs_2d = bb.emit(relax.op.reshape(lhs, [M, K]))

            if len(rhs_shape) > 2:
                rhs_K = tvm.tir.const(1, "int64")
                for s in rhs_shape[:-1]:
                    rhs_K = rhs_K * s
                rhs_K = analyzer.simplify(rhs_K)
                rhs_2d = bb.emit(relax.op.reshape(rhs, [rhs_K, N]))
            else:
                rhs_2d = rhs

            out_2d_call = bb.call_te(te_custom_gemm, lhs_2d, rhs_2d)
            out_2d_var = bb.emit(out_2d_call)
            final_reshape = relax.op.reshape(out_2d_var, out_shape_nd)
            return bb.normalize(final_reshape)

        # =====================================================================
        # 第三部分：构建 Pipeline 并编译
        # =====================================================================
        output = args.output
        if output.suffix in [".tar", ".lib"]:
            system_lib = True
        elif output.suffix in [".so", ".dylib", ".dll"]:
            system_lib = False
        else:
            logger.warning(
                "Unknown output suffix: %s. Assuming shared library.", output.suffix
            )
            system_lib = False

        if compile_type == "opt":
            logger.warning("Compiling with opt")
            custom_relax_pipeline = tvm.transform.Sequential(
                [
                    # fuse_matmul_bias_pass,  # 先将 matmul+add 融合并 lowering
                    relax.transform.LegalizeOps(
                        customize_legalize_map={
                            "relax.matmul": custom_matmul_legalize,  # 剩下的没 bias 的 matmul 走这里
                        }
                    ),
                    relax.transform.DeadCodeElimination(),  # 自动清理被融合后遗弃的原始 matmul 节点
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FoldConstant(),
                    relax.transform.FuseOps(),
                    relax.transform.FuseTIR(),
                ],
            )
            mod = custom_relax_pipeline(mod)
        elif compile_type == "cblas":
            # 使用 cblas.matmul 替换 relax.matmul
            def cblas_matmul_legalize(bb: relax.BlockBuilder, call: relax.Call):
                """
                将 relax.matmul 替换为 tvm.contrib.cblas.matmul 算子
                """
                lhs, rhs = call.args[0], call.args[1]
                lhs_shape = list(lhs.struct_info.shape)
                rhs_shape = list(rhs.struct_info.shape)
                out_shape_nd = list(call.struct_info.shape)

                K, N = lhs_shape[-1], rhs_shape[-1]
                analyzer = tvm.arith.Analyzer()
                M = tvm.tir.const(1, "int64")
                for s in lhs_shape[:-1]:
                    M = M * s
                M = analyzer.simplify(M)

                def te_cblas_gemm(A, B):
                    return cblas.matmul(A, B, transa=False, transb=False)

                lhs_2d = bb.emit(relax.op.reshape(lhs, [M, K]))

                if len(rhs_shape) > 2:
                    rhs_K = tvm.tir.const(1, "int64")
                    for s in rhs_shape[:-1]:
                        rhs_K = rhs_K * s
                    rhs_K = analyzer.simplify(rhs_K)
                    rhs_2d = bb.emit(relax.op.reshape(rhs, [rhs_K, N]))
                else:
                    rhs_2d = rhs

                out_2d_call = bb.call_te(te_cblas_gemm, lhs_2d, rhs_2d)
                out_2d_var = bb.emit(out_2d_call)
                final_reshape = relax.op.reshape(out_2d_var, out_shape_nd)
                return bb.normalize(final_reshape)

            custom_relax_pipeline = tvm.transform.Sequential(
                [
                    relax.transform.LegalizeOps(
                        customize_legalize_map={
                            "relax.matmul": cblas_matmul_legalize,
                        }
                    ),
                    relax.transform.DeadCodeElimination(),
                    relax.transform.AnnotateTIROpPattern(),
                    relax.transform.FoldConstant(),
                    relax.transform.FuseOps(),
                    relax.transform.FuseTIR(),
                ],
            )
            mod = custom_relax_pipeline(mod)
        else:
            pass

        mod = _add_system_lib_prefix(
            mod, args.system_lib_prefix, is_system_lib=system_lib
        )
        relax.build(
            mod,
            target=args.target,
            relax_pipeline=pipeline,
            system_lib=system_lib,
        ).export_library(
            str(output),
        )

    return build


def compile_model(
    model_type: str,
    compile_type: str,
    model_path: str,
    quantization: str,
    target: Device,
    output_dir: str,
    output_lib_path: str,
    debug_dump_dir: Optional[str] = None,
) -> None:
    """
    编译模式
    """
    config_path = os.path.join(output_dir, "mlc-chat-config.json")
    config_path = Path(config_path)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_lib_path = Path(output_lib_path)
    output_lib_path.parent.mkdir(parents=True, exist_ok=True)  # 创建父目录
    build_func = _custom_build_func(compile_type)
    model_type_dict = MODELS[model_type]
    quantization = QUANTIZATION[quantization]
    system_lib_prefix = ""
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    target = tvm.target.Target.from_device(target)
    opt = OptimizationFlags.from_str("O2")
    overrides = ModelConfigOverride.from_str("")
    compile(
        config=config,
        quantization=quantization,
        model_type=model_type_dict,
        target=target,
        opt=opt,
        build_func=build_func,
        system_lib_prefix=system_lib_prefix,
        output=output_lib_path,
        overrides=overrides,
        debug_dump=debug_dump_dir,
    )

    print(f"{model_type}模型编译完成，库文件位于: {output_lib_path}")


def chat_with_llm(
    model_path: str,
    output_lib_path: str,
    target: Device,
    chat_content: str,
    max_tokens: int = None,
    seed: Optional[int] = None,
) -> Tuple[int, int, str]:
    """
    进行对话
    """
    start_time_ns = time.perf_counter_ns()
    engine = MLCEngine(
        model=model_path,  # 配置文件目录
        device=target,
        model_lib=output_lib_path,  # 指定您编译的库文件
    )
    end_time_ns = time.perf_counter_ns()
    load_time = end_time_ns - start_time_ns

    if chat_content == "":
        chat_content = "请介绍一下使用机器学习优化矩阵乘法的技术路线。"
    start_time_ns = time.perf_counter_ns()
    response = engine.chat.completions.create(
        messages=[{"role": "user", "content": chat_content}],
        seed=seed,
        stream=False,
        max_tokens=max_tokens,
    )
    end_time_ns = time.perf_counter_ns()
    infer_time = end_time_ns - start_time_ns
    # logger.debug("response: {}".format(response))
    # logger.debug("dir(response): {}".format(dir(response)))
    rsp_content = ""
    for choice in response.choices:
        rsp_content += choice.message.content
        logger.info(choice.message.content)
    logger.info("验证测试完成!!")
    engine.terminate()
    return load_time, infer_time, rsp_content


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "usage: python compile_llm_models.py <compile_type> <model_name> <seed> <prompt> <prompt_predict_length>"
        )
        exit(1)
    else:
        compile_type = sys.argv[1]
        model_name = sys.argv[2]

        if len(sys.argv) > 4:
            seed = int(sys.argv[3])
            seed = None if seed == 0 else seed
        else:
            seed = None

        if len(sys.argv) > 4:
            chat_content = sys.argv[4]
        else:
            chat_content = "请介绍一下使用机器学习优化矩阵乘法的技术路线。"

        if len(sys.argv) > 5:
            max_tokens = int(sys.argv[5])
        else:
            max_tokens = 10

        if compile_type not in ["opt", "cblas", "origin"]:
            raise ValueError(f"Unknown compile type: {compile_type}")

        if model_name not in [
            "DeepSeek-R1-0528-Qwen3-8B",
            "DeepSeek-Prover-V2-7B",
            "gemma-3-270m",
            "Llama-3.2-3B-Instruct",
            "Ministral-3-3B-Instruct-2512",
            "Qwen3-4B-Instruct-2507",
        ]:
            raise ValueError(f"Unknown model name: {model_name}")

        current_dir = os.path.dirname(__file__)
        output_base_dir = os.path.abspath(
            os.path.join(current_dir, "../../../../", "output")
        )

        model_details = {
            "Qwen3-4B-Instruct-2507": {
                "weight_format": "huggingface-safetensor",
                "model_type": "qwen3",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "Qwen3-4B-Instruct-2507"
                ),
                "output_dir": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Qwen3-4B-Instruct-2507",
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Qwen3-4B-Instruct-2507",
                    "libs",
                    f"qwen3-4b-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
            "DeepSeek-Prover-V2-7B": {
                "weight_format": "huggingface-safetensor",
                "model_type": "llama",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "DeepSeek-Prover-V2-7B"
                ),
                "output_dir": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "DeepSeek-Prover-V2-7B",
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "DeepSeek-Prover-V2-7B",
                    "libs",
                    f"deepseek-prover-v2-7b-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
            "DeepSeek-R1-0528-Qwen3-8B": {
                "weight_format": "huggingface-safetensor",
                "model_type": "qwen3",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "DeepSeek-R1-0528-Qwen3-8B"
                ),
                "output_dir": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "DeepSeek-R1-0528-Qwen3-8B",
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "DeepSeek-R1-0528-Qwen3-8B",
                    "libs",
                    f"deepseek-r1-0528-qwen3-8b-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
            "gemma-3-270m": {
                "weight_format": "huggingface-safetensor",
                "model_type": "gemma3",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "gemma-3-270m"
                ),
                "output_dir": os.path.join(
                    output_base_dir, "model", "compiled", compile_type, "gemma-3-270m"
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "gemma-3-270m",
                    "libs",
                    f"gemma-3-270m-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
            "Llama-3.2-3B-Instruct": {
                "weight_format": "huggingface-safetensor",
                "model_type": "llama",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "Llama-3.2-3B-Instruct"
                ),
                "output_dir": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Llama-3.2-3B-Instruct",
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Llama-3.2-3B-Instruct",
                    "libs",
                    f"llama-3.2-3b-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
            "Ministral-3-3B-Instruct-2512": {
                "weight_format": "huggingface-safetensor",
                "model_type": "ministral3",
                "model_dir": os.path.join(
                    output_base_dir, "model", "origin", "Ministral-3-3B-Instruct-2512"
                ),
                "output_dir": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Ministral-3-3B-Instruct-2512",
                ),
                "quantization": "q0f32",  # 关键：使用 f32 精度
                "output_lib_path": os.path.join(
                    output_base_dir,
                    "model",
                    "compiled",
                    compile_type,
                    "Ministral-3-3B-Instruct-2512",
                    "libs",
                    f"ministral-3-3b-f32-cpu-{compile_type}.so",
                ),
                "conv_template": "chatml",
            },
        }

        (
            model_dir,
            output_dir,
            quantization,
            output_lib_path,
            weight_format,
            model_type,
            conv_template,
        ) = (
            model_details[model_name]["model_dir"],
            model_details[model_name]["output_dir"],
            model_details[model_name]["quantization"],
            model_details[model_name]["output_lib_path"],
            model_details[model_name]["weight_format"],
            model_details[model_name]["model_type"],
            model_details[model_name]["conv_template"],
        )
        Path(output_dir).mkdir(
            parents=True, exist_ok=True
        )  # 目标路径不存在的话，先新建

        target = tvm.cpu()  # 设定目标平台

        init_logs(output_dir)

        envs = [f"{k}: {v}" for k, v in os.environ.items()]
        logger.info(f"environment variables: {', '.join(envs)}")

        start_time_ns = time.perf_counter_ns()
        if not Path(output_lib_path).exists():
            logger.info(f"目标模型{model_type}不存在，先进行编译...")
            convert_qwen3_weights(
                model_type, weight_format, model_dir, quantization, target, output_dir
            )

            gen_qwen3_configs(
                model_type, conv_template, model_dir, quantization, output_dir
            )

            compile_model(
                model_type,
                compile_type,
                model_dir,
                quantization,
                target,
                output_dir,
                output_lib_path,
            )
        end_time_ns = time.perf_counter_ns()
        model_compile_time = end_time_ns - start_time_ns

        load_time, infer_time, rsp_content = chat_with_llm(
            output_dir, output_lib_path, target, chat_content, max_tokens, seed
        )
        log_str = f"[Detail] model_compile_time: {model_compile_time}, model_load_time: {load_time}, infer_time: {infer_time}, rsp_content: {rsp_content}"
        # print(log_str)
        logger.info(log_str)
        logger.info(f"{model_type}模式的MLC LLM 流程测试完成!!")
