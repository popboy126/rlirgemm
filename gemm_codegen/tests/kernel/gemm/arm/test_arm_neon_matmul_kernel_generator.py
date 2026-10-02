# -*- coding: utf-8 -*-

# Author        : huche
# Create Time   : 2025/12/9 16:40
# Contact       : hucheng.lew@qq.com
# File          : test_arm_neon_matmul_kernel_generator.py
# IDE           : PyCharm
# Description   : 测试 ARM NEON 矩阵乘法内核生成器

import os
import shutil
import sys
import tempfile
import unittest
from io import StringIO
from unittest.mock import MagicMock, Mock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

# Add project root to path
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
# print(f"sys.path: {sys.path}")
from gemm_codegen.kernel.gemm.run_gemm_kernel_codegen import main as gemm_codegen_main

# project root dir
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")
)


class TestArmNeonMatMulKernelGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for output files
        # self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(PROJECT_ROOT, "output")
        self.output_test_dir = os.path.join(self.output_dir, "tests")
        self.output_kernel_dir = os.path.join(self.output_dir, "kernels", "arm", "neon")
        self.output_include_dir = os.path.join(self.output_dir, "include")

    # def tearDown(self):
    #     # Remove the temporary directory after tests
    #     shutil.rmtree(self.test_dir)

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr5_nr16_kr8(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 5, 8, 16
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr3_nr7_kr5(self, mock_stdout):
        # Prepare command line arguments
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            "3",
            "--nr",
            "7",
            "--kr",
            "5",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            "mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr1_nr1_kr5(self, mock_stdout):
        # Prepare command line arguments
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            "1",
            "--nr",
            "1",
            "--kr",
            "5",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "2",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            "mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    # @patch('sys.stdout', new_callable=StringIO)
    # def test_generate_neon_f32_kernel_with_mr1_nr64_kr8(self, mock_stdout):
    #     # Prepare command line arguments
    #     test_args = [
    #         os.path.abspath(__file__),
    #         "--platform", "arm",
    #         "--instruction-set", "neon",
    #         "--data-type", "f32",
    #         "--mr", "1", "--nr", "64", "--kr", "8",
    #         "--c-layout", "rowmajor", "--a-layout", "rowmajor" , "--b-layout", "rowmajor",
    #         "--a-packing", "none", "--b-packing", "none",
    #         "--c-prefetch", "L1D", "--a-prefetch", "L2", "--b-prefetch", "L2",
    #         "--edge-optimization",
    #         "--vec-reg-reuse-interval", "4",
    #         "--output-dir", self.output_dir,
    #     ]
    #
    #     with self.assertRaises(Exception):
    #         with patch.object(sys, 'argv', test_args):
    #             gemm_codegen_main()
    #
    #         # 检查kernel文件
    #         expected_kernel_file = os.path.join(self.output_kernel_dir, 'mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr64_kr8.S')
    #         self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr2_nr1_kr5(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 2, 5, 1
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr1_nr5_kr5(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 1, 5, 5
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr1_n7_kr1(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 1, 1, 7
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr2_n8_kr8(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 2, 8, 8
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "none",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr5_nr16_kr8_anp_bfp(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 5, 8, 16
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "fine",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))

    @patch("sys.stdout", new_callable=StringIO)
    def test_generate_neon_f32_kernel_with_mr2_nr4_kr4_anp_bfp(self, mock_stdout):
        # Prepare command line arguments
        mr, kr, nr = 2, 4, 4
        test_args = [
            os.path.abspath(__file__),
            "--platform",
            "arm",
            "--instruction-set",
            "neon",
            "--data-type",
            "f32",
            "--mr",
            f"{mr}",
            "--nr",
            f"{nr}",
            "--kr",
            f"{kr}",
            "--c-layout",
            "rowmajor",
            "--a-layout",
            "rowmajor",
            "--b-layout",
            "rowmajor",
            "--a-packing",
            "none",
            "--b-packing",
            "fine",
            "--c-prefetch",
            "L1D",
            "--a-prefetch",
            "L2",
            "--b-prefetch",
            "L2",
            # "--edge-optimization",
            "--vec-reg-reuse-interval",
            "4",
            "--output-dir",
            self.output_dir,
        ]

        with patch.object(sys, "argv", test_args):
            gemm_codegen_main()

        # 检查kernel文件
        expected_kernel_file = os.path.join(
            self.output_kernel_dir,
            f"mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr{mr}_nr{nr}_kr{kr}.S",
        )
        self.assertTrue(os.path.isfile(expected_kernel_file))
