include(FetchContent)
set(GTEST_VERSION_TAG "v1.17.0" CACHE STRING "GoogleTest tag")
FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG "${GTEST_VERSION_TAG}")
FetchContent_MakeAvailable(googletest)
set(gtest_SOURCE_DIR "${googletest_SOURCE_DIR}/googletest")
enable_testing()
include(GoogleTest)
