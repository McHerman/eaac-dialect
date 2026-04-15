add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp
  MemoryAlloc.cpp
  MemRefLivenessAnalysis.cpp
  LocalStaging.cpp
  LowerCopyToDma.cpp
  EncodeDependencies.cpp
  FindAsyncDependency.cpp
  InsertRequire.cpp
  LowerAsyncToSemaphore.cpp

  DEPENDS
  EAACPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRAsyncDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRAnalysis
)
