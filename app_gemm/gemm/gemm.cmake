
# 首地址
set(GEMM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/gemm")
set(GEMM_WRAPPER_HOME "${GEMM_HOME}/wrappers")

include(${GEMM_HOME}/kernels/kernels.cmake)

# The generated opt_gemm executable below is self-contained.  The ml_gemm
# shared/static libraries are a separate OpenBLAS-backed integration used by
# external applications, so do not make them a prerequisite of the core build.
if (RLIRGEMM_HAS_OPENBLAS)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(ML_GEMM_SOURCES
            ${GEMM_HOME}/ml_gemm.cpp
            ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS}
            ${ADD_KERNELS_ARM_NEON_IMPLEMENTATIONS}
        )

        add_library(ml_gemm_static STATIC ${ML_GEMM_SOURCES})
        target_include_directories(ml_gemm_static
                PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
                PRIVATE ${ADD_KERNELS_ARM_NEON_HOME}
                PRIVATE ${GEMM_WRAPPER_HOME}
                PRIVATE ${GEMM_HOME}
                PRIVATE ${OPENBLAS_INSTALL_DIR}/include
                PRIVATE ${OPENBLAS_HOME})

        add_library(ml_gemm_shared SHARED ${ML_GEMM_SOURCES})
        target_include_directories(ml_gemm_shared
                PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}
                PRIVATE ${ADD_KERNELS_ARM_NEON_HOME}
                PRIVATE ${GEMM_WRAPPER_HOME}
                PRIVATE ${GEMM_HOME}
                PRIVATE ${OPENBLAS_INSTALL_DIR}/include
                PRIVATE ${OPENBLAS_HOME})
    else ()
        set(ML_GEMM_SOURCES ${GEMM_HOME}/ml_gemm.cpp)

        add_library(ml_gemm_static STATIC ${ML_GEMM_SOURCES})
        target_include_directories(ml_gemm_static
                PRIVATE ${GEMM_WRAPPER_HOME}
                PRIVATE ${GEMM_HOME}
                PRIVATE ${OPENBLAS_INSTALL_DIR}/include
                PRIVATE ${OPENBLAS_HOME})

        add_library(ml_gemm_shared SHARED ${ML_GEMM_SOURCES})
        target_include_directories(ml_gemm_shared
                PRIVATE ${GEMM_WRAPPER_HOME}
                PRIVATE ${GEMM_HOME}
                PRIVATE ${OPENBLAS_INSTALL_DIR}/include
                PRIVATE ${OPENBLAS_HOME})
    endif()

    add_dependencies(ml_gemm_static OpenBLASLib)
    add_dependencies(ml_gemm_shared OpenBLASLib)
    target_link_libraries(ml_gemm_static PUBLIC OpenMP::OpenMP_CXX PRIVATE ${OpenBlas_LIBRARY})
    target_link_libraries(ml_gemm_shared PUBLIC OpenMP::OpenMP_CXX PRIVATE ${OpenBlas_LIBRARY})
    set_target_properties(ml_gemm_static PROPERTIES OUTPUT_NAME "ml_gemm")
    set_target_properties(ml_gemm_shared PROPERTIES OUTPUT_NAME "ml_gemm")
else()
    message(STATUS "Skipping ml_gemm_static/ml_gemm_shared; no OpenBLAS prefix was supplied")
endif()

IF (CMAKE_BUILD_TYPE MATCHES "Debug"
        OR CMAKE_BUILD_TYPE MATCHES "None")
    MESSAGE(STATUS "This project's CMAKE_BUILD_TYPE is Debug")
#    set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
#    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address")
ELSEIF (CMAKE_BUILD_TYPE MATCHES "Release")
    MESSAGE(STATUS "This project's CMAKE_BUILD_TYPE is Release")
ELSE ()
    MESSAGE(STATUS "This trails project's CMAKE_BUILD_TYPE = " ${CMAKE_BUILD_TYPE})
ENDIF ()


############下面是优化算法
if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_executable(gemm_by_opt_gemm
            ${BASE_HOME}/gemm_main.cpp
            ${GEMM_KERNELS_ARM_NEON_IMPLEMENTATIONS}
            ${ADD_KERNELS_ARM_NEON_IMPLEMENTATIONS}
    )

    target_include_directories(gemm_by_opt_gemm
            PRIVATE ${GEMM_KERNELS_ARM_NEON_HOME}  # 加入neon内核的头文件
            PRIVATE ${ADD_KERNELS_ARM_NEON_HOME}
            PRIVATE ${GEMM_WRAPPER_HOME}
            PRIVATE ${GEMM_HOME}
            PRIVATE ${BASE_HOME}
    )
else ()
    add_executable(gemm_by_opt_gemm
            ${BASE_HOME}/gemm_main.cpp
    )

    target_include_directories(gemm_by_opt_gemm
            PRIVATE ${GEMM_WRAPPER_HOME}
            PRIVATE ${GEMM_HOME}
            PRIVATE ${BASE_HOME}
    )
endif ()

target_link_libraries(gemm_by_opt_gemm
        PRIVATE OpenMP::OpenMP_CXX)

target_compile_definitions(gemm_by_opt_gemm
        PRIVATE _TRAIL_LABEL=opt_gemm
        PRIVATE BENCH_OPT_O0=${BENCH_OPT_O0_VAL}
        PRIVATE CHECK_GEMM_RESULT_VAL=${CHECK_GEMM_RESULT_VAL}
        PRIVATE GEMM_BY_COMM_HEADER="by_opt_gemm.h"
)

# 开启asan
enable_asan(gemm_by_opt_gemm)
