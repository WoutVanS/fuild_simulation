cmake_minimum_required(VERSION 3.27)
project(fuild_simulation CUDA)

set(CMAKE_CUDA_STANDARD 14)

include(FetchContent)
set(BUILD_SHARED_LIBS  OFF)
FETCHCONTENT_DECLARE(SFML GIT_REPOSITORY https://github.com/SFML/SFML.git GIT_TAG 2.5.1)
FETCHCONTENT_MAKEAVAILABLE(SFML)


# Define the executable target
add_executable(fuild_simulation GPU.cu)

target_link_libraries(fuild_simulation sfml-graphics)

# Set properties for the target
set_target_properties(fuild_simulation PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)