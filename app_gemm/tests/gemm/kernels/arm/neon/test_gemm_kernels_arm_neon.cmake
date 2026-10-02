

set(TESTS_GEMM_KERNELS_ARM_NEON_HOME "${TESTS_GEMM_KERNELS_ARM_HOME}/neon")



add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr3_nr7_kr5.unittest)



add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr1_kr5.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr1_kr5.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr5_kr5.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr1_nr7_kr1.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr2_nr8_kr8.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr8_kr16.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr8_kr16.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr8_kr16.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr8_kr16.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bnp_mr5_nr8_kr16.unittest)



# 下面是A不打包，B采用细粒度打包的测试
add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr8_kr16.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr8_kr16.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr8_kr16.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr8_kr16.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr5_nr8_kr16.unittest)


add_executable(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.unittest
        ${TESTS_GEMM_KERNELS_ARM_NEON_HOME}/test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.cpp
        ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.unittest
        PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
        PRIVATE ${GEMM_WRAPPER_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_mm_kernel_arm_neon_fp32_cr_ar_br_anp_bfp_mr2_nr4_kr4.unittest)
