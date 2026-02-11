//===- CollectAllocDealloc.cpp - Collect alloc/dealloc and print liveness -===//
//
// Experimental pass to collect memref alloc/dealloc operations and
// print liveness information for memory allocation analysis.
//
//===----------------------------------------------------------------------===//

#include "eaac/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_COLLECTALLOCDEALLOC
#include "eaac/Passes.h.inc"

namespace {

/// Assigns a unique "time" to each operation for liveness computation.
class OperationNumbering {
public:
  void compute(Operation *rootOp) {
    int64_t time = 0;
    rootOp->walk([&](Operation *op) { opToTime[op] = time++; });
  }

  int64_t getTime(Operation *op) const {
    auto it = opToTime.find(op);
    if (it != opToTime.end())
      return it->second;
    return -1;
  }

private:
  llvm::DenseMap<Operation *, int64_t> opToTime;
};

/// Computes the size in bytes for a memref type.
uint64_t getMemRefSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape())
    return 0; // Dynamic shapes not supported

  int64_t numElements = 1;
  for (int64_t dim : type.getShape())
    numElements *= dim;

  Type elementType = type.getElementType();
  uint64_t elementBits = 0;

  if (elementType.isIntOrFloat()) {
    elementBits = elementType.getIntOrFloatBitWidth();
  } else {
    elementBits = 32; // Default
  }

  return numElements * ((elementBits + 7) / 8);
}

//===----------------------------------------------------------------------===//
// CollectAllocDealloc Pass
//===----------------------------------------------------------------------===//

class CollectAllocDeallocPass
    : public impl::CollectAllocDeallocBase<CollectAllocDeallocPass> {
public:
  using CollectAllocDeallocBase::CollectAllocDeallocBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Separate SmallVectors for alloc and dealloc operations
    llvm::SmallVector<memref::AllocOp, 16> allocOps;
    llvm::SmallVector<memref::DeallocOp, 16> deallocOps;

    // Collect all alloc operations
    module.walk([&](memref::AllocOp allocOp) { allocOps.push_back(allocOp); });

    // Collect all dealloc operations
    module.walk(
        [&](memref::DeallocOp deallocOp) { deallocOps.push_back(deallocOp); });

    llvm::errs() << "========================================\n";
    llvm::errs() << "Collected Memory Operations\n";
    llvm::errs() << "========================================\n\n";

    llvm::errs() << "Found " << allocOps.size() << " alloc operations\n";
    llvm::errs() << "Found " << deallocOps.size() << " dealloc operations\n\n";

    // Process each function for liveness analysis
    module.walk([&](func::FuncOp funcOp) {
      llvm::errs() << "----------------------------------------\n";
      llvm::errs() << "Function: " << funcOp.getName() << "\n";
      llvm::errs() << "----------------------------------------\n\n";

      // Compute operation numbering
      OperationNumbering numbering;
      numbering.compute(funcOp);

      // Compute MLIR Liveness analysis
      Liveness liveness(funcOp);

      // Collect allocs in this function
      llvm::SmallVector<memref::AllocOp, 8> funcAllocs;
      funcOp.walk(
          [&](memref::AllocOp allocOp) { funcAllocs.push_back(allocOp); });

      // Collect deallocs in this function
      llvm::SmallVector<memref::DeallocOp, 8> funcDeallocs;
      funcOp.walk(
          [&](memref::DeallocOp deallocOp) { funcDeallocs.push_back(deallocOp); });

      llvm::errs() << "Allocations in function:\n";
      for (auto allocOp : funcAllocs) {
        MemRefType type = allocOp.getType();
        uint64_t size = getMemRefSizeInBytes(type);
        int64_t allocTime = numbering.getTime(allocOp);

        llvm::errs() << "  Alloc at time " << allocTime << ":\n";
        llvm::errs() << "    Type: " << type << "\n";
        llvm::errs() << "    Size: " << size << " bytes\n";

        // Find all users and their times
        int64_t lastUseTime = allocTime;
        int64_t deallocTime = -1;

        llvm::errs() << "    Users:\n";
        for (Operation *user : allocOp.getResult().getUsers()) {
          int64_t userTime = numbering.getTime(user);
          llvm::errs() << "      - " << user->getName() << " at time "
                       << userTime << "\n";

          if (isa<memref::DeallocOp>(user)) {
            deallocTime = userTime;
          }
          lastUseTime = std::max(lastUseTime, userTime);
        }

        llvm::errs() << "    Liveness interval: [" << allocTime << ", "
                     << lastUseTime << "]\n";
        if (deallocTime >= 0) {
          llvm::errs() << "    Deallocated at time: " << deallocTime << "\n";
        } else {
          llvm::errs() << "    No explicit deallocation found\n";
        }

        // Use MLIR Liveness to check which blocks the value is live in
        Value memrefValue = allocOp.getResult();
        llvm::errs() << "    Live in blocks: ";
        bool first = true;
        for (Block &block : funcOp.getBody()) {
          const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
          if (blockInfo && blockInfo->isLiveIn(memrefValue)) {
            if (!first)
              llvm::errs() << ", ";
            llvm::errs() << "^bb" << &block;
            first = false;
          }
        }
        if (first) {
          llvm::errs() << "(none - defined in entry block)";
        }
        llvm::errs() << "\n\n";
      }

      llvm::errs() << "Deallocations in function:\n";
      for (auto deallocOp : funcDeallocs) {
        int64_t deallocTime = numbering.getTime(deallocOp);
        Value memref = deallocOp.getMemref();

        llvm::errs() << "  Dealloc at time " << deallocTime << ":\n";

        // Try to find the defining alloc
        if (auto definingOp = memref.getDefiningOp()) {
          int64_t defTime = numbering.getTime(definingOp);
          llvm::errs() << "    Deallocating value defined at time " << defTime
                       << "\n";
          llvm::errs() << "    Defined by: " << definingOp->getName() << "\n";
        } else {
          llvm::errs() << "    Deallocating block argument\n";
        }
        llvm::errs() << "\n";
      }
    });

    llvm::errs() << "========================================\n";
    llvm::errs() << "End of Liveness Analysis\n";
    llvm::errs() << "========================================\n";
  }
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Pass Creation Function
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createCollectAllocDeallocPass() {
  return std::make_unique<CollectAllocDeallocPass>();
}

} // namespace eaac
} // namespace mlir
