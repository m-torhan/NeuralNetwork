cmake_minimum_required(VERSION 3.16)

project(NeuralNetwork LANGUAGES C CXX ASM ASM_NASM)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ./${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./${CMAKE_BUILD_TYPE})

set(CMAKE_ASM_NASM_COMPILER nasm)
if(NOT WIN32)
    set(CMAKE_ASM_NASM_COMPILE_OBJECT "<CMAKE_ASM_NASM_COMPILER> <INCLUDES> <FLAGS> -f elf32 -o <OBJECT> <SOURCE>")
    set(CMAKE_FIND_ROOT_PATH   /usr/i486-linux-gnu)
endif(NOT WIN32)

set(CMAKE_C_FLAGS "-m32")
if("Release" STREQUAL ${CMAKE_BUILD_TYPE})
    set(CMAKE_CXX_FLAGS "-g -O3 -Wall -m32")
else()
    set(CMAKE_CXX_FLAGS "-g -O0 -Wall -m32")
endif()

include(CPM.cmake)

if("Release" STREQUAL ${CMAKE_BUILD_TYPE})
    CPMAddPackage(
        NAME googletest
        GITHUB_REPOSITORY google/googletest
        GIT_TAG release-1.11.0
        OPTIONS -DBUILD_TESTING=OFF
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    if(WIN32)
        message("WARNING: Google benchmark not supported on Windows")
    else(WIN32)
        CPMAddPackage(
            NAME benchmark
            GITHUB_REPOSITORY google/benchmark
            GIT_TAG v1.6.1
            OPTIONS -DBUILD_TESTING=OFF
        )
    endif(WIN32)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(WIN32)
    set(OS "Win")
    add_definitions( -DWIN )
else(WIN32)
    set(OS "Linux")
endif(WIN32)

if("SSE" STREQUAL "${CMAKE_BUILD_MODE}")
    add_definitions( -DSSE )
endif()

set(SOURCE_DIR ../src)
if("Release" STREQUAL ${CMAKE_BUILD_TYPE})
    set(UNIT_TEST_DIR ../tests/unit_tests)
    set(PERFORMANCE_TEST_DIR ../tests/performance_tests)
endif()
set(APP_MNIST_DIR ../applications/mnist)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/..")

add_subdirectory(${SOURCE_DIR} ${SOURCE_DIR}/Build/${OS})
if("Release" STREQUAL ${CMAKE_BUILD_TYPE})
    add_subdirectory(${UNIT_TEST_DIR} ${UNIT_TEST_DIR}/Build/${OS})
    if(NOT WIN32)
        add_subdirectory(${PERFORMANCE_TEST_DIR} ${PERFORMANCE_TEST_DIR}/Build/${OS})
    endif(NOT WIN32)
endif()
add_subdirectory(${APP_MNIST_DIR} ${APP_MNIST_DIR}/Build/${OS})