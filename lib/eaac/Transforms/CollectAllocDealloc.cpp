//===- CollectAllocDealloc.cpp - Print liveness intervals for allocations -===//
//
// Pass to collect memref allocations and print their liveness intervals
// using MLIR's built-in Liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "eaac/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_COLLECTALLOCDEALLOC
#include "eaac/Passes.h.inc"

namespace {

/// Computes the size in bytes for a memref type.
uint64_t getMemRefSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;

  int64_t numElements = 1;
  for (int64_t dim : type.getShape())
    numElements *= dim;

  Type elementType = type.getElementType();
  uint64_t elementBits = elementType.isIntOrFloat()
      ? elementType.getIntOrFloatBitWidth()
      : 32;

  return numElements * ((elementBits + 7) / 8);
}

class CollectAllocDeallocPass
    : public impl::CollectAllocDeallocBase<CollectAllocDeallocPass> {
public:
  using CollectAllocDeallocBase::CollectAllocDeallocBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](func::FuncOp funcOp) {

      // Step 1: Number all operations
      llvm::DenseMap<Operation *, int64_t> opTime;
      int64_t time = 0;
      funcOp.walk([&](Operation *op) { opTime[op] = time++; });

      // Step 2: Compute liveness using MLIR's analysis
      Liveness liveness(funcOp);

      // Step 3: Process each allocation
      funcOp.walk([&](memref::AllocOp allocOp) {
        Value memref = allocOp.getResult();
        Operation *startOp = allocOp;
        Operation *endOp = allocOp;

        // Find the last use across all blocks using Liveness
        for (Block &block : funcOp.getBody()) {
          const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
          if (!blockInfo)
            continue;

          if (Operation *end = blockInfo->getEndOperation(memref, startOp)) {
            int64_t endTime = opTime[end];
            int64_t currentEndTime = opTime[endOp];
            if (endTime > currentEndTime)
              endOp = end;
          }
        }

        // Look up times for start and end operations
        int64_t startTime = opTime[startOp];
        int64_t endTime = opTime[endOp];

        MemRefType type = allocOp.getType();
        uint64_t size = getMemRefSizeInBytes(type);

        llvm::errs() << funcOp.getName() << ": " << type
                     << " (" << size << " bytes)"
                     << " live: [" << startTime << ", " << endTime << "]\n";
      });
    });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createCollectAllocDeallocPass() {
  return std::make_unique<CollectAllocDeallocPass>();
}

} // namespace eaac
} // namespace mlir
