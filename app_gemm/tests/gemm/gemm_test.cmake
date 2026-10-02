
set(TESTS_GEMM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/tests/gemm")

include(${TESTS_GEMM_HOME}/kernels/test_gemm_kernels.cmake)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(
            test_opt_gemm.unittest
            ${TESTS_GEMM_HOME}/test_opt_gemm.cpp
            ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS}
            ${ADD_KERNELS_ARM_NEON_IMPLEMENTATIONS}
    )


    target_link_libraries(
            test_opt_gemm.unittest
            PRIVATE gtest
            PRIVATE gtest_main
            PUBLIC OpenMP::OpenMP_CXX
    )


    target_include_directories(
            test_opt_gemm.unittest
            PRIVATE ${BASE_HOME}
            PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
            PRIVATE ${ADD_KERNELS_ARM_NEON_HOME}
            PRIVATE ${GEMM_WRAPPER_HOME}
            PRIVATE ${GEMM_HOME}
            PRIVATE ${gtest_SOURCE_DIR}/include
    )
else()
    add_executable(
            test_opt_gemm.unittest
            ${TESTS_GEMM_HOME}/test_opt_gemm.cpp
    )


    target_link_libraries(
            test_opt_gemm.unittest
            PRIVATE gtest
            PRIVATE gtest_main
            PUBLIC OpenMP::OpenMP_CXX
    )


    target_include_directories(
            test_opt_gemm.unittest
            PRIVATE ${BASE_HOME}
            PRIVATE ${GEMM_WRAPPER_HOME}
            PRIVATE ${GEMM_HOME}
            PRIVATE ${gtest_SOURCE_DIR}/include
    )
endif ()
gtest_discover_tests(test_opt_gemm.unittest)
