# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/3 16:05
# Contact       : hucheng.lew@qq.com
# File          : gen_tvm_python_build_toml.py
# IDE           : PyCharm
# Description   : 获取pyhon生成的toml配置文件

import sys
import codecs
import os
import json

def gen_tvm_cmake_build_config(openblas_home: str, ml_gemm_inc_path: str, ml_gemm_lib_path: str, output_file_path: str):
    """
    生成cmake构建配置
    """
    config_context = f"""set(USE_LLVM "/usr/bin/llvm-config-18")
#set(USE_LLVM ON)
set(USE_CUDA   OFF)
set(USE_METAL  OFF)
set(USE_VULKAN OFF)
set(USE_OPENCL OFF)
set(USE_CUBLAS OFF)
set(USE_CUDNN  OFF)
set(USE_CUTLASS OFF)
set(USE_BLAS openblas)
set(BLAS_INCLUDE_DIR {openblas_home}/include)
set(BLAS_LIBRARY {openblas_home}/lib/libopenblas.so)
set(USE_GRAPH_EXECUTOR ON)
set(USE_CPP_RPC ON)
# set(BUILD_DUMMY_LIBTVM ON)
set(TVM_BUILD_PYTHON_MODULE ON)
set(BACKTRACE_ON_SEGFAULT ON)
set(USE_CUSTOM_GEMM_INCLUDE {ml_gemm_inc_path})
set(USE_CUSTOM_GEMM_LIB {ml_gemm_lib_path})

# 调试信息
set(CMAKE_BUILD_TYPE RelWithDebInfo)
#set(TVM_DEBUG_WITH_ABI_CHANGE ON)
#set(TVM_LOG_BEFORE_THROW ON)
#set(TVM_FFI_ATTACH_DEBUG_SYMBOLS ON)
#set(HIDE_PRIVATE_SYMBOLS OFF)
"""

    with codecs.open(output_file_path, "w", encoding="utf-8") as f:
        f.write(config_context)
    print(f"tvm compilation config file '{output_file_path}' generated successfully.")


def patching_tvm_pyproject_toml(tvm_home: str, openblas_home: str, ml_gemm_inc_path: str, ml_gemm_lib_path: str):
    """
    修改pyproject.toml以保证python打包过程可顺利进行
    """
    cmake_build_args = [
        "-DUSE_LLVM=/usr/bin/llvm-config-18",
        "-DUSE_CUDA=OFF",
        "-DUSE_METAL=OFF",
        "-DUSE_VULKAN=OFF",
        "-DUSE_OPENCL=OFF",
        "-DUSE_CUBLAS=OFF",
        "-DUSE_CUDNN=OFF",
        "-DUSE_CUTLASS=OFF",
        "-DUSE_BLAS=openblas",
        f"-DBLAS_INCLUDE_DIR={openblas_home}/include",
        f"-DBLAS_LIBRARY={openblas_home}/lib/libopenblas.so",
        "-DUSE_GRAPH_EXECUTOR=ON",
        "-DUSE_CPP_RPC=ON",
        # "-DBUILD_DUMMY_LIBTVM=ON",
        "-DTVM_BUILD_PYTHON_MODULE=ON",
        # "-DBACKTRACE_ON_SEGFAULT=ON",
        f"-DUSE_CUSTOM_GEMM_INCLUDE={ml_gemm_inc_path}",
        f"-DUSE_CUSTOM_GEMM_LIB={ml_gemm_lib_path}",
    ]

    cmake_build_args_str = json.dumps(cmake_build_args, indent=2)
    pyproject_toml_path = os.path.join(tvm_home, "pyproject.toml")
    pyproject_contents = []
    with codecs.open(pyproject_toml_path, "r", encoding="utf-8") as f:
        for i in f:
            if i.startswith("cmake.args = "):
                pyproject_contents.append(f"cmake.args = {cmake_build_args_str}")
            else:
                pyproject_contents.append(i.rstrip())

    with codecs.open(pyproject_toml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pyproject_contents))

    print(f"tvm wheel config file '{pyproject_toml_path}' generated successfully.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <build_type> ....")

    build_type = sys.argv[1].strip()

    if build_type == "compile":
        if len(sys.argv) < 6:
            print(f"Usage: python {sys.argv[0]} compile <openblas_home> <ml_gemm_inc_path> <ml_gemm_lib_path> <output_file_path>")
        openblas_home = sys.argv[2]
        ml_gemm_inc_path = sys.argv[3]
        ml_gemm_lib_path = sys.argv[4]
        output_file_path = sys.argv[5]
        gen_tvm_cmake_build_config(openblas_home, ml_gemm_inc_path, ml_gemm_lib_path, output_file_path)
    elif build_type == "wheel":
        if len(sys.argv) < 6:
            print(f"Usage: python {sys.argv[0]} wheel <tvm_home> <openblas_home> <ml_gemm_inc_path> <ml_gemm_lib_path>")
        tvm_home = sys.argv[2]
        openblas_home = sys.argv[3]
        ml_gemm_inc_path = sys.argv[4]
        ml_gemm_lib_path = sys.argv[5]
        patching_tvm_pyproject_toml(tvm_home, openblas_home, ml_gemm_inc_path, ml_gemm_lib_path)
