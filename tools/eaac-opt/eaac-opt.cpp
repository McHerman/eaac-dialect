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

  // Register EAAC transform ops as a Transform dialect extension
  mlir::eaac::registerEAACTransformDialectExtension(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "EAAC optimizer driver\n", registry));
}
