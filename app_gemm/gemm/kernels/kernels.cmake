
set(GEMM_KERNELS_HOME "${GEMM_HOME}/kernels")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    include(${GEMM_KERNELS_HOME}/arm/arm.cmake)
endif ()
