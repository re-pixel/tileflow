cmake_minimum_required(VERSION 3.15)
project(mini_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Runtime core library (static)
add_library(mini_runtime_core STATIC
    src/runtime/memory/arena.cpp
    src/runtime/memory/tensors.cpp
    src/runtime/kernels/matmul_ref.cpp
    src/runtime/engine/engine.cpp
)

target_include_directories(mini_runtime_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(mini_runtime_core PRIVATE
    -Wall -Wextra -Wpedantic
)

# pybind11 module
pybind11_add_module(mini_runtime 
    src/bindings/bindings.cpp
)

target_link_libraries(mini_runtime PRIVATE mini_runtime_core)

# Optional: C++ tests using CTest
enable_testing()

# Find if we have a test framework (optional)
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_kernels tests/test_kernels.cpp)
    target_link_libraries(test_kernels mini_runtime_core GTest::gtest_main)
    add_test(NAME KernelTests COMMAND test_kernels)
endif()

# Installation (for development)
install(TARGETS mini_runtime DESTINATION .)
