cmake_minimum_required(VERSION 3.22.2)

project(NeuralNetwork)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./${CMAKE_BUILD_TYPE})

include(CPM.cmake)

CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    GIT_TAG release-1.11.0
    OPTIONS -DBUILD_TESTING=OFF
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(SOURCE_DIR ../src)
set(TEST_DIR ../tests)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

add_subdirectory(${SOURCE_DIR} ${SOURCE_DIR}/Build)
add_subdirectory(${TEST_DIR} ${TEST_DIR}/Build)