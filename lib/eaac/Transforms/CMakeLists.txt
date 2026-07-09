add_mlir_library(EAACTransforms
  CollectAllocDealloc.cpp
  MemoryAlloc.cpp
  MemRefLivenessAnalysis.cpp
  StreamingChannelAnalysis.cpp
  LocalStaging.cpp
  LowerCopyToDma.cpp
  EncodeDependencies.cpp
  FindAsyncDependency.cpp
  FindAliasDependency.cpp
  CorrectBroadcast.cpp
  InsertRequire.cpp
  LowerAsyncToSemaphore.cpp
  InsertLoadStore.cpp
  AssignSemaphoreAddresses.cpp
  LegalizeForEaacHw.cpp
  RiscvKernelToFunction.cpp

  DEPENDS
  EAACPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRAsyncDialect
  MLIRLinalgDialect
  MLIRArithDialect
  MLIRAnalysis
)
