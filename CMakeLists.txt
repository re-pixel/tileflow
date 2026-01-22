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

# Check for AVX2 and FMA support
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
check_cxx_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
    set(HAS_AVX2_FMA TRUE)
    message(STATUS "AVX2 and FMA support detected - enabling optimized kernels")
else()
    set(HAS_AVX2_FMA FALSE)
    message(STATUS "AVX2/FMA not supported - using reference kernels only")
endif()

# Find pybind11
find_package(pybind11 REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Kernel source files
set(KERNEL_SOURCES
    src/runtime/kernels/matmul_ref.cpp
    src/runtime/kernels/dispatcher.cpp
)

# Add AVX2 kernel if supported
if(HAS_AVX2_FMA)
    list(APPEND KERNEL_SOURCES src/runtime/kernels/matmul_avx2.cpp)
endif()

# Runtime core library (static)
add_library(mini_runtime_core STATIC
    src/runtime/memory/arena.cpp
    src/runtime/memory/tensors.cpp
    src/runtime/engine/engine.cpp
    ${KERNEL_SOURCES}
)

target_include_directories(mini_runtime_core PUBLIC 
    ${CMAKE_SOURCE_DIR}/include
)

target_compile_options(mini_runtime_core PRIVATE
    -Wall -Wextra -Wpedantic
)

# Compile AVX2 kernel with appropriate flags
if(HAS_AVX2_FMA)
    set_source_files_properties(
        src/runtime/kernels/matmul_avx2.cpp
        src/runtime/kernels/dispatcher.cpp
        PROPERTIES COMPILE_FLAGS "-mavx2 -mfma"
    )
endif()

# pybind11 module
pybind11_add_module(mini_runtime 
    src/bindings/bindings.cpp
)

target_link_libraries(mini_runtime PRIVATE mini_runtime_core)

# Add AVX2 flags to the pybind11 module if available
if(HAS_AVX2_FMA)
    target_compile_options(mini_runtime PRIVATE -mavx2 -mfma)
endif()

# Optional: C++ tests using CTest
enable_testing()

# Find if we have a test framework (optional)
find_package(GTest QUIET)
if(GTest_FOUND)
    add_executable(test_kernels tests/test_kernels.cpp)
    target_link_libraries(test_kernels mini_runtime_core GTest::gtest_main)
    if(HAS_AVX2_FMA)
        target_compile_options(test_kernels PRIVATE -mavx2 -mfma)
    endif()
    add_test(NAME KernelTests COMMAND test_kernels)
endif()

# Installation (for development)
install(TARGETS mini_runtime DESTINATION .)
