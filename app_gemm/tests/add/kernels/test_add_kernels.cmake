

set(TESTS_ADD_KERNELS_HOME "${TESTS_ADD_HOME}/kernels")

if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    include(${TESTS_ADD_KERNELS_HOME}/arm/test_add_kernels_arm.cmake)
endif ()
