set(UNIT_TEST_BINARY ${CMAKE_PROJECT_NAME}_unit_tests)
file(GLOB_RECURSE UNIT_TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

if(NOT MSVC)
    add_definitions(-DUSE_MOCKS)
endif()

enable_testing()

add_executable(${UNIT_TEST_BINARY} ${UNIT_TEST_SOURCES})
add_test(NAME ${UNIT_TEST_BINARY} COMMAND ${UNIT_TEST_BINARY})

target_link_libraries(${UNIT_TEST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gtest)
target_link_libraries(${UNIT_TEST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib gmock)