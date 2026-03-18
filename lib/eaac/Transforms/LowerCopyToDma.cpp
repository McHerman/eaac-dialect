//===- LowerCopyToDma.cpp - Lower memref.copy to DMA start/wait pairs -===//
//
// Pass to replace memref.copy operations with eaac.dma_start/dma_wait pairs.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "lower-copy-to-dma"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LOWERCOPYTODMA
#include "eaac/Passes.h.inc"

namespace {

class LowerCopyToDmaPass
    : public impl::LowerCopyToDmaBase<LowerCopyToDmaPass> {
public:
  using LowerCopyToDmaBase::LowerCopyToDmaBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Collect all memref.copy ops first to avoid iterator invalidation.
    llvm::SmallVector<memref::CopyOp> copyOps;
    module.walk([&](memref::CopyOp op) { copyOps.push_back(op); });

    for (auto copyOp : copyOps) {
      OpBuilder builder(copyOp);
      Location loc = copyOp.getLoc();

      Value src = copyOp.getSource();
      Value dst = copyOp.getTarget();

      // Replace memref.copy with dma_start + dma_wait.
      auto dmaStart = DmaStartOp::create(
          builder, loc, builder.getIndexType(), src, dst);
      DmaWaitOp::create(builder, loc, dmaStart.getToken());

      copyOp.erase();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLowerCopyToDmaPass() {
  return std::make_unique<LowerCopyToDmaPass>();
}

} // namespace eaac
} // namespace mlir
