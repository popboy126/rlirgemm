
set(TESTS_HOME "${CMAKE_CURRENT_SOURCE_DIR}/tests")

include(${TESTS_HOME}/test_common.cmake)

include(${TESTS_HOME}/add/add_test.cmake)
include(${TESTS_HOME}/gemm/gemm_test.cmake)
