//===- MemRefLivenessAnalysis.cpp - MemRef lifetime analysis -----*- C++ -*-===//
//
// Computes live intervals for all memref allocations in a module.
//
//===----------------------------------------------------------------------===//

#include "eaac/MemRefLivenessAnalysis.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "local-staging"

namespace mlir {
namespace eaac {

MemRefLivenessAnalysis::MemRefLivenessAnalysis(Operation *op) {
  auto module = cast<ModuleOp>(op);

  module.walk([&](func::FuncOp funcOp) {
    // Step 1: Number all operations
    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });

    // Step 2: Compute liveness using MLIR's analysis
    Liveness liveness(funcOp);

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

      // Collect all uses of this memref
      llvm::SmallVector<MemRefUse> memRefUses;
      for (Operation *user : memref.getUsers()) {
        if (opTime.count(user)) {
          memRefUses.push_back({user, opTime[user]});
        }
      }
      // Sort uses chronologically by time
      llvm::sort(memRefUses, [](const MemRefUse &a, const MemRefUse &b) {
        return a.time < b.time;
      });

      // Extract use times for the LiveInterval
      llvm::SmallVector<int64_t> useTimes;
      for (const auto &use : memRefUses) {
        useTimes.push_back(use.time);
      }

      // Create a unique ID for this allocation
      std::string id = std::to_string(startTime);

      // Compute buffer size (number of elements)
      int64_t size = 1;
      auto memrefType = llvm::cast<MemRefType>(memref.getType());
      for (int64_t dim : memrefType.getShape()) {
        if (dim != ShapedType::kDynamic)
          size *= dim;
      }

      // Create and store the LiveInterval
      LiveInterval interval(id, size, startTime, endTime, std::move(useTimes),
                            memref);
      intervals.push_back(std::move(interval));

      LLVM_DEBUG({
        llvm::dbgs() << "MemRef allocation at time " << startTime << ": "
                     << memref << "\n  Lifetime: [" << startTime << ", "
                     << endTime << "] Size: " << size << " Uses:";
        for (const auto &use : memRefUses)
          llvm::dbgs() << " " << use.time << "("
                       << use.op->getName().getStringRef() << ")";
        llvm::dbgs() << "\n";
      });
    });
  });
}

} // namespace eaac
} // namespace mlir
