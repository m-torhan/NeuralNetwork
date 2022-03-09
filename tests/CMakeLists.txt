set(TEST_BINARY ${CMAKE_PROJECT_NAME}_tests)
file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

if(NOT MSVC)
    add_definitions(-DUSE_MOCKS)
endif()

enable_testing()

add_executable(${TEST_BINARY} ${TEST_SOURCES})
add_test(NAME ${TEST_BINARY} COMMAND ${TEST_BINARY})

target_link_libraries(${TEST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest)
target_link_libraries(${TEST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gmock)