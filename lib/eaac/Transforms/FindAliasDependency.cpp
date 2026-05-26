//===- FindAliasDependency.cpp - Token edges for aliased memrefs ---------===//
//
// Adds async.token edges that arise from allocator-induced memref aliasing,
// i.e. cases where LocalStaging assigned two SSA-distinct memrefs to the
// same physical {tier, offset} range.
//
// Runs after FindAsyncDependency. The existing SSA-level token graph is
// correct under the single-writer invariant; this pass fills in the alias-
// induced edges that the invariant cannot express.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "find-alias-dependency"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_FINDALIASDEPENDENCY
#include "eaac/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Op-time numbering
//===----------------------------------------------------------------------===//

/// Assigns a monotonically increasing integer to every op in `funcOp` in
/// walk order. Use the resulting map to compare temporal order between any
/// two ops in the function ("does X come before Y?").
///
/// Mirrors the numbering done by MemRefLivenessAnalysis so that the times
/// produced here can be cross-referenced with LiveInterval starts/ends.
static llvm::DenseMap<Operation *, int64_t>
buildOpTimeMap(func::FuncOp funcOp) {
  llvm::DenseMap<Operation *, int64_t> opTime;
  int64_t time = 0;
  funcOp.walk([&](Operation *op) { opTime[op] = time++; });
  return opTime;
}

//===----------------------------------------------------------------------===//
// Transitivity check
//===----------------------------------------------------------------------===//

/// True iff `later` already transitively depends on `earlier` via existing
/// async.token edges. Walks the token def-use graph backwards from `later`.
///
/// Pruned by op-time: any producer with op-time strictly less than
/// `earlier`'s op-time cannot reach `earlier` via further backward edges
/// (token edges only go from consumer to earlier producer), so it's not
/// worth expanding.
///
/// If `earlier` has no entry in `opTime`, this returns false defensively.
/// Callers should ensure both ops are in the same function.
static bool transitivelyOrdered(
    async::ExecuteOp later, async::ExecuteOp earlier,
    const llvm::DenseMap<Operation *, int64_t> &opTime) {
  if (later == earlier)
    return true;

  auto earlierIt = opTime.find(earlier);
  if (earlierIt == opTime.end())
    return false;
  const int64_t earlierTime = earlierIt->second;

  llvm::SmallPtrSet<Operation *, 32> visited;
  llvm::SmallVector<async::ExecuteOp, 16> work;
  work.push_back(later);

  while (!work.empty()) {
    async::ExecuteOp cur = work.pop_back_val();
    for (Value tok : cur.getDependencies()) {
      auto producer = tok.getDefiningOp<async::ExecuteOp>();
      if (!producer)
        continue;
      if (producer == earlier)
        return true;
      auto it = opTime.find(producer);
      if (it == opTime.end() || it->second < earlierTime)
        continue; // can't reach `earlier` from here
      if (visited.insert(producer).second)
        work.push_back(producer);
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class FindAliasDependencyPass
    : public impl::FindAliasDependencyBase<FindAliasDependencyPass> {
public:
  using FindAliasDependencyBase::FindAliasDependencyBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  void processFunction(func::FuncOp funcOp) {
    llvm::DenseMap<Operation *, int64_t> opTime = buildOpTimeMap(funcOp);

    // TODO: detect alias pairs from eaac.offset / eaac.tier / size + lifespans
    //       and emit token edges for the ones not transitively ordered.
    //
    // Sketch:
    //   1. Collect every memref-defining op carrying eaac.offset + eaac.tier.
    //   2. Group by tier; sort by op-time.
    //   3. For each pair (A, B) with A before B in op-time, same tier, and
    //      overlapping [offset, offset+size) ranges:
    //        a. Find lastReader(A) and firstWriter(B) as async.execute ops.
    //        b. If transitivelyOrdered(firstWriter(B), lastReader(A), opTime)
    //           → skip (already covered).
    //        c. Else if op_time(firstWriter(B)) - op_time(lastReader(A)) > W
    //           → emit a fence at firstWriter(B) (TODO: fence op).
    //        d. Otherwise append lastReader(A)'s token to firstWriter(B)'s
    //           dependency list (rebuild the async.execute op since its
    //           operand list is part of its construction).
    (void)opTime;
    (void)funcOp;
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createFindAliasDependencyPass() {
  return std::make_unique<FindAliasDependencyPass>();
}

} // namespace eaac
} // namespace mlir
