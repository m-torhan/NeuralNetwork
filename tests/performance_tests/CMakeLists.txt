set(PERFORMANCE_TEST_BINARY ${CMAKE_PROJECT_NAME}_performance_tests)
file(GLOB_RECURSE PERFORMANCE_TEST_SOURCES LIST_DIRECTORIES false *.h *.cpp)

enable_testing()

add_executable(${PERFORMANCE_TEST_BINARY} ${PERFORMANCE_TEST_SOURCES})
add_test(NAME ${PERFORMANCE_TEST_BINARY} COMMAND ${PERFORMANCE_TEST_BINARY})

target_link_libraries(${PERFORMANCE_TEST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib benchmark)