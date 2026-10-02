

set(TESTS_ADD_KERNELS_ARM_NEON_HOME "${TESTS_ADD_KERNELS_ARM_HOME}/neon")


add_executable(
        test_add_kernel_arm_neon_fp32_r_mr2_nr16_kr2.unittest
        ${TESTS_ADD_KERNELS_ARM_NEON_HOME}/test_add_kernel_arm_neon_fp32_r_mr2_nr16_kr2.cpp
        ${ADD_KERNELS_ARM_NEON_IMPLEMENTATIONS_FOR_TESTS}
)


target_link_libraries(
        test_add_kernel_arm_neon_fp32_r_mr2_nr16_kr2.unittest
        PRIVATE gtest
        PRIVATE gtest_main
        PUBLIC OpenMP::OpenMP_CXX
)


target_include_directories(
        test_add_kernel_arm_neon_fp32_r_mr2_nr16_kr2.unittest
        PRIVATE ${ADD_KERNELS_ARM_NEON_HOME}
        PRIVATE ${gtest_SOURCE_DIR}/include
)
gtest_discover_tests(test_add_kernel_arm_neon_fp32_r_mr2_nr16_kr2.unittest)
