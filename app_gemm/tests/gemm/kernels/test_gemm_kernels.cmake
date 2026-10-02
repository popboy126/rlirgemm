
set(TESTS_GEMM_KERNELS_HOME "${TESTS_GEMM_HOME}/kernels")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    include(${TESTS_GEMM_KERNELS_HOME}/arm/test_gemm_kernels_arm.cmake)
endif ()
