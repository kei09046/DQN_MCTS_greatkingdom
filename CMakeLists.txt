cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)
cmake_policy(SET CMP0144 NEW)

# Set paths
set(CMAKE_PREFIX_PATH "/home/me/libtorch")
set(CMAKE_PREFIX_PATH "home/me/boost_1_89_0" ${CMAKE_PREFIX_PATH})
set(CAFFE2_USE_CUDNN ON)
set(USE_CUSPARSELT OFF)
find_package(Torch REQUIRED)
find_package(Boost REQUIRED)

# Compiler and linker flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pg")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

# Debug mode definitions
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(_GLIBCXX_DEBUG)
endif()

# Define the measurement flag as an option
# option(MEASURE_TIME "Enable time measurement" ON)
# For training, set trans_table = ON. for very high playout turn it off.
option(TRANS_TABLE "enable transition table" ON) 

if (MEASURE_TIME)
    add_compile_definitions(measureTime)
endif()
if (TRANS_TABLE)
    add_compile_definitions(transTable)
endif()

# Create the executable
add_executable(train 
    main.cpp 
    gamerules.cpp 
    hash.cpp
    evalcache.cpp
    memorypool.cpp 
    PMCTS.cpp 
    policyvalue.cpp 
    random.cpp 
    rotation.cpp
    elo.cpp 
    consts.cpp
    train.cpp
)

# Link against LibTorch
target_link_libraries(train "${TORCH_LIBRARIES}")
target_link_libraries(train Boost::headers)

# Include directories
target_include_directories(train PRIVATE 
    "/home/me/libtorch/include" 
    "/home/me/libtorch/include/torch/csrc/api/include"
)

# Set C++ standard
set_property(TARGET train PROPERTY CXX_STANDARD 20)

# Ensure build type is not overridden unexpectedly
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()
