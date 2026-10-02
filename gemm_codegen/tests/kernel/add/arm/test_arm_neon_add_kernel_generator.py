# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/29 09:16
# Contact       : hucheng.lew@qq.com
# File          : test_arm_neon_add_kernel_generator.py
# IDE           : PyCharm
# Description   : neon指令实现的累加内核生成器测试代码

import unittest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from gemm_codegen.kernel.add.run_add_kernel_codegen import main as add_codegen_main

# project root dir
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..'))


class TestArmNeonMatMulKernelGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output files
        # self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(PROJECT_ROOT, 'output')
        self.output_test_dir = os.path.join(self.output_dir, 'tests')
        self.output_kernel_dir = os.path.join(self.output_dir, 'kernels', "arm", "neon")
        self.output_include_dir = os.path.join(self.output_dir, 'include')

    # def tearDown(self):
    #     # Remove the temporary directory after tests
    #     shutil.rmtree(self.test_dir)


    @patch('sys.stdout', new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr2_nr16_kr2(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 2, 2, 16
        test_args = [
            os.path.abspath(__file__),
            "--platform", "arm",
            "--instruction-set", "neon",
            "--data-type", "f32",
            "--mr", f"{mr}", "--nr", f"{nr}", "--kr", f"{kr}",
            "--layout", "rowmajor",
            "--c-prefetch", "L1D", "--partial-sum-prefetch", "L1D",
            "--edge-optimization",
            "--vec-reg-reuse-interval", "4",
            "--output-dir", self.output_dir,
        ]

        with patch.object(sys, 'argv', test_args):
            add_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(self.output_kernel_dir, f'add_kernel_arm_neon_fp32_r_mr{mr}_nr{nr}_kr{kr}.S')
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch('sys.stdout', new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr4_nr12_kr1(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 2, 2, 16
        test_args = [
            os.path.abspath(__file__),
            "--platform", "arm",
            "--instruction-set", "neon",
            "--data-type", "f32",
            "--mr", f"{mr}", "--nr", f"{nr}", "--kr", f"{kr}",
            "--layout", "rowmajor",
            "--c-prefetch", "L1D", "--partial-sum-prefetch", "L1D",
            "--edge-optimization",
            "--vec-reg-reuse-interval", "4",
            "--output-dir", self.output_dir,
        ]

        with patch.object(sys, 'argv', test_args):
            add_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(self.output_kernel_dir, f'add_kernel_arm_neon_fp32_r_mr{mr}_nr{nr}_kr{kr}.S')
        self.assertTrue(os.path.isfile(expected_kernel_file))

