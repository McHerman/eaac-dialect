//===- eaac-opt.cpp - EAAC optimizer driver -------------------------------===//
//
// Main entry point for the EAAC dialect optimizer.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "eaac/Dialect.h"
#include "eaac/Passes.h"
#include "eaac/Transform/EAACTransformOps.h"

int main(int argc, char **argv) {
  // Register MLIR core passes (like bufferization)
  mlir::registerAllPasses();

  // Register EAAC passes
  mlir::eaac::registerEAACPasses();

  mlir::DialectRegistry registry;
  
  // Register all standard MLIR dialects
  mlir::registerAllDialects(registry);
  
  // Register all dialect extensions (needed for inlining, etc.)
  mlir::registerAllExtensions(registry);
  
  // Register our EAAC dialect
  registry.insert<mlir::eaac::EAACDialect>();

  // Attach ScheduleInterface external models to ops that support scheduling
  mlir::eaac::registerScheduleOpInterfaceExternalModels(registry);

  // Register EAAC transform ops as a Transform dialect extension
  mlir::eaac::registerEAACTransformDialectExtension(registry);

  // Needed by SplitLLVMFromEAAC to translate its extracted LLVM module.
  // Must happen here (before the context is built / passes run), not lazily
  // inside a pass, since MLIRContext forbids registry mutation once the
  // PassManager's multi-threaded execution context is active. Both the
  // LLVM dialect (for the llvm.* ops) and the builtin dialect (for the
  // builtin.module root itself) need a translation interface registered.
  mlir::registerLLVMDialectTranslation(registry);
  mlir::registerBuiltinDialectTranslation(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "EAAC optimizer driver\n", registry));
}
