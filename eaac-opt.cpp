//===- eaac-opt.cpp - EAAC optimizer driver -------------------------------===//
//
// Main entry point for the EAAC dialect optimizer.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "eaac/Dialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register all standard MLIR dialects
  mlir::registerAllDialects(registry);
  
  // Register our EAAC dialect
  registry.insert<mlir::eaac::EAACDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "EAAC optimizer driver\n", registry));
}
