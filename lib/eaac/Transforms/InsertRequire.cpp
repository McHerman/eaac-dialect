//===- InsertRequire.cpp - Pair async tokens with memrefs ----------------===//
//
// Inserts eaac.require ops to explicitly bind each async.token dependency
// to the memref it guards, replacing bare memref uses with the require result.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "insert-require"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_INSERTREQUIRE
#include "eaac/Passes.h.inc"

namespace {

/// Returns the memref Value that `op` writes to, or nullptr if none found.
Value findWrittenMemref(Operation *op) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return nullptr;
  llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    Value val = effect.getValue();
    if (val && isa<MemRefType>(val.getType()))
      return val;
  }
  return nullptr;
}


class InsertRequirePass
    : public impl::InsertRequireBase<InsertRequirePass> {
public:
  using InsertRequireBase::InsertRequireBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  void processFunction(func::FuncOp funcOp) {
    funcOp.walk([&](async::ExecuteOp executeOp) {
      llvm::DenseMap<Value, Value> tokenToMemref;

      for (auto token : executeOp.getDependencies()) {
        auto producerExec = token.getDefiningOp<async::ExecuteOp>();
        if (!producerExec)
          continue;

        // Walk the producer's body to find which memref it writes to.
        producerExec.getBody()->walk([&](Operation *innerOp) {
          Value memref = findWrittenMemref(innerOp);
          if (memref)
            tokenToMemref[token] = memref;
        });
      }

      // Insert eaac.require at the start of the execute body.
      OpBuilder builder(executeOp.getBody(), executeOp.getBody()->begin());
      for (auto &[token, memref] : tokenToMemref) {
        RequireOp::create(builder, executeOp.getLoc(),
                          memref.getType(), token, memref);
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createInsertRequirePass() {
  return std::make_unique<InsertRequirePass>();
}

} // namespace eaac
} // namespace mlir
