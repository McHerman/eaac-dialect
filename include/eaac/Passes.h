//===- Passes.h - EAAC dialect passes ---------------------------*- C++ -*-===//
//
// Header file for EAAC dialect passes.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_PASSES_H
#define EAAC_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace eaac {

//===----------------------------------------------------------------------===//
// Pass Declarations (Generated)
//===----------------------------------------------------------------------===//

#define GEN_PASS_DECL
#include "eaac/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Creates a pass to collect alloc/dealloc operations and print liveness info.
std::unique_ptr<Pass> createCollectAllocDeallocPass();

/// Creates pass to statically allocate memory
std::unique_ptr<Pass> createMemoryAllocPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "eaac/Passes.h.inc"

} // namespace eaac
} // namespace mlir

#endif // EAAC_PASSES_H
