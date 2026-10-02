# --- 函数定义 ---
# 功能: 在 /proc/cpuinfo 中检测指定的 CPU 型号字符串
# 参数:
#   CPU_STRING_TO_FIND (IN):  要查找的 CPU 型号字符串
#   RESULT_VAR (OUT):         用于存储结果的变量名 (TRUE 或 FALSE)
#
function(DetectCpuModel CPU_STRING_TO_FIND RESULT_VAR)
    # 检查输入参数是否为空
    if(NOT CPU_STRING_TO_FIND)
        message(FATAL_ERROR "DetectCpuModel: CPU_STRING_TO_FIND cannot be empty.")
    endif()
    message(STATUS "Detecting CPU model: searching for '${CPU_STRING_TO_FIND}' in /proc/cpuinfo...")
    # 初始化一个局部变量来存储结果
    set(_is_found FALSE)
    # 1. 检查 /proc/cpuinfo 文件是否存在
    if(EXISTS "/proc/cpuinfo")
        # 2. 读取文件内容到列表变量中
        file(STRINGS "/proc/cpuinfo" CPUINFO_LIST)
        # 3. 遍历并匹配
        foreach(LINE ${CPUINFO_LIST})
            string(FIND "${LINE}" "${CPU_STRING_TO_FIND}" FOUND_INDEX)
            if(NOT ${FOUND_INDEX} EQUAL -1)
                message(STATUS "DetectCpuModel: Found target CPU string '${CPU_STRING_TO_FIND}'.")
                set(_is_found TRUE)
                break() # 找到后立即退出循环
            endif()
        endforeach()
    else()
        message(WARNING "DetectCpuModel: /proc/cpuinfo not found. Cannot detect CPU on this system.")
    endif()
    # 4. 将结果传递回调用者
    # CMake 3.18+ 可以简化为: return(PROPAGATE ${RESULT_VAR})
    # 为了兼容 3.17，我们使用 set(... PARENT_SCOPE)
    set(${RESULT_VAR} ${_is_found} PARENT_SCOPE)
endfunction()
