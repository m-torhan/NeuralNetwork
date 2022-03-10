set(BINARY ${CMAKE_PROJECT_NAME}_run)
set(LIBRARY ${CMAKE_PROJECT_NAME}_lib)

file(GLOB_RECURSE SOURCES LIST_DIRECTORIES true *.h *.cpp)
if("SSE" STREQUAL "${CMAKE_BUILD_MODE}")
    file(GLOB_RECURSE SOURCES_ASM LIST_DIRECTORIES true *.asm)
    list(APPEND SOURCES ${SOURCES_ASM})
endif()
list(REMOVE_ITEM SOURCES "main.cpp")
set(SOURCES ${SOURCES})

add_executable(${BINARY} main.cpp)

add_library(${CMAKE_PROJECT_NAME}_lib STATIC ${SOURCES})
target_link_libraries(${BINARY} ${LIBRARY})