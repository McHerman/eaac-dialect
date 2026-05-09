//===- eaac-translate.cpp - EAAC translation driver -----------------------===//
//
// Main entry point for translating EAAC MLIR to FlatBuffer binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"

#include "eaac/Dialect.h"
#include "eaac/Target/EAACToFlatbuffer.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard dialects (needed for parsing input).
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::DLTIDialect>();

  // Register the EAAC dialect.
  registry.insert<mlir::eaac::EAACDialect>();

  // Register EAAC translation targets.
  mlir::eaac::registerEAACToFlatbufferTranslation(registry);

  return mlir::failed(mlir::mlirTranslateMain(argc, argv, "EAAC translation driver\n"));
}
