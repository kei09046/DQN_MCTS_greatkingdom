cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)
cmake_policy(SET CMP0144 NEW)

# Set paths
set(CMAKE_PREFIX_PATH "/home/me/libtorch")
set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)
find_package(Torch REQUIRED)

# Compiler and linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

# Create the executable
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(play ${SRC})

# Link against LibTorch
target_link_libraries(play "${TORCH_LIBRARIES}")

# Include directories
target_include_directories(play PRIVATE 
    "/home/me/libtorch/include" 
    "/home/me/libtorch/include/torch/csrc/api/include"
)

# Set C++ standard
set_property(TARGET play PROPERTY CXX_STANDARD 20)

add_compile_options(-fsanitize=address -g)
add_link_options(-fsanitize=address)

# Ensure build type is not overridden unexpectedly
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
