# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/3/3 14:25
# Contact       : hucheng.lew@qq.com
# File          : gen_mlc_llm_config.py
# IDE           : PyCharm
# Description   : 生成mlc-llm的配置文件


import sys
import codecs
import os
import json
from typing import List

def gen_mlc_llm_config_file(tvm_source_path: str, output_file_path: str = "config.cmake"):
    """
    生成TVM的环境变量文件

    Parameters
    ----------
    tvm_source_path : str
        TVM的源码安装路径
    output_file : str
        输出的环境变量文件路径
    """
    cmake_config_content = [
        f"set(TVM_SOURCE_DIR {tvm_source_path})",
        "set(CMAKE_BUILD_TYPE RelWithDebInfo)",
        "set(USE_CUDA OFF)",
        "set(USE_CUTLASS OFF)",
        "set(USE_CUBLAS OFF)",
        "set(USE_ROCM OFF)",
        "set(USE_VULKAN OFF)",
        "set(USE_METAL OFF)",
        "set(USE_OPENCL OFF)",
        "set(USE_OPENCL_ENABLE_HOST_PTR OFF)"
    ]
    with open(output_file_path, "w") as f:
        f.write("\n".join(cmake_config_content))
    print(f"MLC LLM compilation config file '{output_file_path}' generated successfully.")

def patching_mlc_llm_pyproject_toml(mlc_llm_home: str) -> None:
    """
    限制并行编译的线程数
    """

    pyproject_toml_path = os.path.join(mlc_llm_home, "pyproject.toml")
    pyproject_contents = []
    with codecs.open(pyproject_toml_path, "r", encoding="utf-8") as f:
        for i in f:
            if i.startswith("cmake.args = "):
                cmake_args: List[str] = json.loads(i.lstrip("cmake.args = "))
                pyproject_contents.append(f"cmake.args = {json.dumps(cmake_args, indent=2)}")
            else:
                pyproject_contents.append(i.rstrip())

    with codecs.open(pyproject_toml_path, "w", encoding="utf-8") as f:
        f.write("\n".join(pyproject_contents))
    print(f"MLC LLM wheel config file '{pyproject_toml_path}' generated successfully.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: python {sys.argv[0]} <mlc_llm_home> <tvm_source_path> <output_file_path>")
    else:
        mlc_llm_home = sys.argv[1]
        tvm_source_path = sys.argv[2]
        output_file_path = sys.argv[3]
        patching_mlc_llm_pyproject_toml(mlc_llm_home)
        gen_mlc_llm_config_file(tvm_source_path, output_file_path)
