# Optional integration: provide an already checked-out TVM tree.
set(TVM_SOURCE_DIR "" CACHE PATH "External TVM source directory")
if(NOT TVM_SOURCE_DIR OR NOT EXISTS "${TVM_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ENABLE_TVM requires -DTVM_SOURCE_DIR=/path/to/tvm (use the documented paper commit).")
endif()
add_custom_target(tvm_external_integration
    COMMAND "${CMAKE_COMMAND}" -E echo "TVM source: ${TVM_SOURCE_DIR}"
    VERBATIM)
