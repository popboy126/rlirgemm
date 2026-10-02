include(CheckCXXCompilerFlag)
include(CMakePushCheckState)
# 定义一个函数来开启 AddressSanitizer
function(enable_asan target_name)
    # 检查编译器是否支持该标志
    check_cxx_compiler_flag("-fsanitize=address" SUPPORTS_ASAN)

    if(SUPPORTS_ASAN)
        # 仅在 Debug 模式下开启
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            message(STATUS "Enabling AddressSanitizer for target ${target_name}")

            # 添加编译选项
            target_compile_options(${target_name} PRIVATE
                    -fsanitize=address
                    -fno-omit-frame-pointer  # 可选：获取更清晰的堆栈信息
                    -g                       # 确保包含调试符号
            )

            # 添加链接选项
            target_link_options(${target_name} PRIVATE
                    -fsanitize=address
            )
        endif()
    else()
        message(WARNING "Compiler does not support -fsanitize=address for target ${target_name}")
    endif()
endfunction()
