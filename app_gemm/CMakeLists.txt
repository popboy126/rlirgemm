cmake_minimum_required(VERSION 3.18)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(app_gemm
        VERSION 1.0.0)

include(ProcessorCount)

# 获取核心数并存入变量 N_CORES
ProcessorCount(N_CORES)
# 如果检测失败（N_CORES为0），则设置一个默认值
IF(N_CORES EQUAL 0)
    message(STATUS "Could not detect processor count, defaulting to 1.")
    set(N_CORES 1)
    set(MAX_N_CORES_LOAD 1)
ELSEIF (N_CORES GREATER 8)
    set(N_CORES 8)
    set(MAX_N_CORES_LOAD 7)
ELSE()
    math(EXPR MAX_N_CORES_LOAD "${N_CORES} - 1")
ENDIF()

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake;${CMAKE_MODULE_PATH}")
include (cmake/utils.cmake)
include(cmake/env_params.cmake)
include(cmake/cpu_detect.cmake)
include(cmake/sanitizers.cmake)


if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    # 检查CPU型号
    DetectCpuModel("Phytium,FT-2000+/64" IS_PHYTIUM_CPU)
    if (IS_PHYTIUM_CPU)
        set(CPU_TYPE "FT2000+")
    else()
        DetectCpuModel("0x48" IS_KUNPENG_CPU)
        if (IS_KUNPENG_CPU)
            set(CPU_TYPE "KunPeng920")
        else ()
            DetectCpuModel("0x41" IS_GRAVITON2_CPU)
            if (IS_GRAVITON2_CPU)
                set(CPU_TYPE "Graviton2")
            else()
                message(FATAL_ERROR "Unsupported platform for aarch64")
            endif()
        endif()
    endif()
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

enable_language(ASM)

find_package(OpenMP REQUIRED)

option(BENCH_OPT_LEVEL_O0 "Force the compiler use O0 opt level to compile performance_benchmark function!" OFF)

if (BENCH_OPT_LEVEL_O0)
    MESSAGE(STATUS "BENCH_OPT_LEVEL_O0 is ON, will use O0 opt level to compile performance_benchmark function!")
    set(BENCH_OPT_O0_VAL 1)
else ()
    MESSAGE(STATUS "BENCH_OPT_LEVEL_O0 is OFF, will use default opt level to compile performance_benchmark function!")
    set(BENCH_OPT_O0_VAL 0)
endif()

option(CHECK_GEMM_RESULT "Build with the result check logic!" OFF)
if (CHECK_GEMM_RESULT)
    MESSAGE(STATUS "CHECK_GEMM_RESULT is ON, will build with the result check logic!")
    set(CHECK_GEMM_RESULT_VAL 1)
else ()
    MESSAGE(STATUS "CHECK_GEMM_RESULT is OFF, will build without the result check logic!")
    set(CHECK_GEMM_RESULT_VAL 0)
endif()


set(BASE_HOME "${CMAKE_CURRENT_SOURCE_DIR}/base")
set(GEMM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/gemm")

# Dependencies and optional external baseline adapters.
include(baselines/baselines.cmake)

# 优化方案
include(add/add.cmake)
include(gemm/gemm.cmake)

# Optional TVM/MLC integrations are disabled by default.
include(applications/applications.cmake)

# 正确性测试（可在无网络的基线适配配置阶段关闭；默认仍开启）
option(BUILD_GEMM_TESTS "Fetch and build GoogleTest-based correctness tests" ON)
if (BUILD_GEMM_TESTS)
    include(tests/tests.cmake)
else()
    message(STATUS "BUILD_GEMM_TESTS is OFF; correctness tests are not configured")
endif()
