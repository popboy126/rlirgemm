# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2026/2/4 19:23
# Contact       : hucheng.lew@qq.com
# File          : tvm_env_file_generator.py
# IDE           : PyCharm
# Description   : 生成tvm的环境变量文件

import sys

def gen_tvm_env_file(tvm_path: str, ml_gemm_lib_path: str, output_file_path: str = "tvm_env.sh"):
    """
    生成TVM的环境变量文件

    Parameters
    ----------
    tvm_path : str
        TVM的安装路径
    output_file : str
        输出的环境变量文件路径
    """
    with open(output_file_path, "w") as f:
        f.write(f"export TVM_HOME={tvm_path}\n")
        f.write('export TVM_LOG_DEBUG="DEFAULT=-1"\n')
        f.write("export PYTHONPATH=${TVM_HOME}/python:${PYTHONPATH}\n")
        f.write("export TVM_LIBRARY_PATH=${TVM_HOME}/build\n")
        # f.write("export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/13/libasan.so\n")
        # f.write('export ASAN_OPTIONS="verify_asan_link_order=0:detect_leaks=0:halt_on_error=0:abort_on_error=0"\n')
        f.write("export LD_LIBRARY_PATH=${TVM_LIBRARY_PATH}:${TVM_LIBRARY_PATH}/lib:%s:${LD_LIBRARY_PATH}\n" % ml_gemm_lib_path)
    print(f"TVM environment file '{output_file_path}' generated successfully.")



if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python tvm_env_file_generator.py <tvm_path> <ml_gemm_lib_path> <output_file_path>")
    else:
        tvm_path = sys.argv[1]
        ml_gemm_lib_path = sys.argv[2]
        output_file_path = sys.argv[3]
        gen_tvm_env_file(tvm_path, ml_gemm_lib_path, output_file_path)
