set(APP_MNIST_VAE_BINARY ${CMAKE_PROJECT_NAME}_app_mnist_vae_run)

file(GLOB_RECURSE APP_MNIST_VAE_SOURCES LIST_DIRECTORIES true *.h *.cpp)
list(REMOVE_ITEM APP_MNIST_VAE_SOURCES "mnist_vae_main.cpp")
set(APP_MNIST_VAE_SOURCES ${SOURCES})

add_executable(${APP_MNIST_VAE_BINARY} mnist_vae_main.cpp)
target_link_libraries(${APP_MNIST_VAE_BINARY} PUBLIC ${CMAKE_PROJECT_NAME}_lib)