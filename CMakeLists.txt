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

# Add minimalloc for static memory allocation
# EXCLUDE_FROM_ALL ensures only minimalloc_core is built (not tests/CLI)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../minimalloc minimalloc EXCLUDE_FROM_ALL)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include)

set(LLVM_LINK_COMPONENTS
  Core
  Support
)

add_llvm_executable(eaac-opt
  eaac-opt.cpp
  mlir/Dialect.cpp
  mlir/CollectAllocDealloc.cpp
  mlir/StaticMemoryAllocation.cpp

  DEPENDS
  EAACIncGen
  EAACPassIncGen
)

target_link_libraries(eaac-opt
  PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    minimalloc_core
)
