//===- EncodeDependencies.cpp - Wrap ops in async.execute regions ------===//
//
// Pass to wrap DMA and compute operations in async.execute regions.
// Dependency tokens are not threaded here — a later pass handles that.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "encode-dependencies"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_ENCODEDEPENDENCIES
#include "eaac/Passes.h.inc"

namespace {

/// Wrap a single operation in an async.execute region.
async::ExecuteOp wrapInAsyncExecute(OpBuilder &builder, Operation *op) {
  Location loc = op->getLoc();

  auto executeOp = async::ExecuteOp::create(
      builder, loc, /*resultTypes=*/TypeRange{},
      /*dependencies=*/ValueRange{}, /*operands=*/ValueRange{});

  Block *body = executeOp.getBody();

  // Remove the auto-generated yield.
  if (!body->empty())
    body->back().erase();

  // Move the op into the region.
  op->moveBefore(body, body->end());

  // Add async.yield terminator.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(op);
  async::YieldOp::create(builder, loc, /*operands=*/ValueRange{});

  return executeOp;
}

class EncodeDependenciesPass
    : public impl::EncodeDependenciesBase<EncodeDependenciesPass> {
public:
  using EncodeDependenciesBase::EncodeDependenciesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  void processFunction(func::FuncOp funcOp) {
    // Collect all ops to wrap first to avoid iterator invalidation.
    llvm::SmallVector<Operation *> targetOps;
    funcOp.walk([&](Operation *op) {
      if (isa<DmaStartOp>(op) ||
          (isa<linalg::LinalgOp>(op) && !isa<linalg::FillOp>(op)))
        targetOps.push_back(op);
    });

    for (auto *op : targetOps) {
      OpBuilder builder(op);
      wrapInAsyncExecute(builder, op);
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createEncodeDependenciesPass() {
  return std::make_unique<EncodeDependenciesPass>();
}

} // namespace eaac
} // namespace mlir
