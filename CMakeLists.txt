cmake_minimum_required(VERSION 3.25.1)
project(eaac-dialect LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Minimalloc
set(MINIMALLOC_DIR "${PROJECT_SOURCE_DIR}/../minimalloc")
include_directories(${MINIMALLOC_DIR}/src)
include_directories(${MINIMALLOC_DIR}/external/abseil-cpp)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
link_directories(${MINIMALLOC_DIR}/build)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/status)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/strings)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/time)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/base)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/numeric)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/debugging)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/synchronization)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/hash)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/container)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/crc)
link_directories(${MINIMALLOC_DIR}/build/external/abseil-cpp/absl/profiling)
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
