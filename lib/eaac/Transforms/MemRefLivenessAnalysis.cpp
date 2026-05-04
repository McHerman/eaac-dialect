//===- MemRefLivenessAnalysis.cpp - MemRef lifetime analysis -----*- C++ -*-===//
//
// Computes live intervals for all memref allocations in a module.
//
//===----------------------------------------------------------------------===//

#include "eaac/MemRefLivenessAnalysis.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AsmState.h"
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

    // Compute a LiveInterval for any memref-producing result.
    auto addInterval = [&](Value memref, Operation *defOp) {
      Operation *startOp = defOp;
      Operation *endOp = defOp;

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

      int64_t startTime = opTime[startOp];
      int64_t endTime = opTime[endOp] + 1;

      llvm::SmallVector<MemRefUse> memRefUses;
      for (Operation *user : memref.getUsers()) {
        if (opTime.count(user) && !isa<memref::DeallocOp>(user)) {
          memRefUses.push_back({user, opTime[user]});
        }
      }
      llvm::sort(memRefUses, [](const MemRefUse &a, const MemRefUse &b) {
        return a.time < b.time;
      });

      std::string id;
      {
        AsmState state(funcOp);
        llvm::raw_string_ostream os(id);
        memref.printAsOperand(os, state);
      }

      auto memrefType = llvm::cast<MemRefType>(memref.getType());
      mlir::Type elementType = memrefType.getElementType();
      int64_t bytes = elementType.getIntOrFloatBitWidth() / 8;
      int64_t size = 1;
      for (int64_t dim : memrefType.getShape()) {
        if (dim != ShapedType::kDynamic)
          size *= dim;
      }
      size *= bytes;

      LLVM_DEBUG({
        llvm::dbgs() << "MemRef allocation at time " << startTime << ": "
                     << memref << "\n  Lifetime: [" << startTime << ", "
                     << endTime << "] Size: " << size << " Uses:";
        for (const auto &use : memRefUses)
          llvm::dbgs() << " " << use.time << "("
                       << use.op->getName().getStringRef() << ")";
        llvm::dbgs() << "\n";
      });

      LiveInterval interval(id, size, startTime, endTime, std::move(memRefUses),
                            memref);
      intervals.push_back(std::move(interval));
    };

    // Walk all ops and collect intervals for any that produce memref results.
    funcOp.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        if (isa<MemRefType>(result.getType())) {
          addInterval(result, op);
        }
      }
    });
  });
}

} // namespace eaac
} // namespace mlir
