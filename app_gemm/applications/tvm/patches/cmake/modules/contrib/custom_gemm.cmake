

if(USE_CUSTOM_GEMM_INCLUDE AND USE_CUSTOM_GEMM_LIB)
    message(STATUS "Using custom GEMM headers in ${USE_CUSTOM_GEMM_INCLUDE} and library in " ${USE_CUSTOM_GEMM_LIB})
    include_directories(SYSTEM ${USE_CUSTOM_GEMM_INCLUDE})
    list(APPEND TVM_RUNTIME_LINKER_LIBS ${USE_CUSTOM_GEMM_LIB})
    list(APPEND RUNTIME_SRCS src/runtime/contrib/custom_gemm_lib/custom_gemm_runtime.cc)
endif ()
