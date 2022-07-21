set(APP_MNIST_BINARY ${CMAKE_PROJECT_NAME}_app_mnist_run)

file(GLOB_RECURSE APP_MNIST_SOURCES LIST_DIRECTORIES true *.h *.cpp)
list(REMOVE_ITEM APP_MNIST_SOURCES "mnist_main.cpp")
set(APP_MNIST_SOURCES ${SOURCES})

add_executable(${APP_MNIST_BINARY} mnist_main.cpp)
target_link_libraries(${APP_MNIST_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib)