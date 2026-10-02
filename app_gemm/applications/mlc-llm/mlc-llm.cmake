# Optional integration: provide an already checked-out MLC LLM tree.
set(MLC_LLM_SOURCE_DIR "" CACHE PATH "External MLC LLM source directory")
if(NOT MLC_LLM_SOURCE_DIR OR NOT EXISTS "${MLC_LLM_SOURCE_DIR}/python")
    message(FATAL_ERROR "ENABLE_MLC_LLM requires -DMLC_LLM_SOURCE_DIR=/path/to/mlc-llm.")
endif()
add_custom_target(mlc_llm_external_integration
    COMMAND "${CMAKE_COMMAND}" -E echo "MLC LLM source: ${MLC_LLM_SOURCE_DIR}"
    VERBATIM)
