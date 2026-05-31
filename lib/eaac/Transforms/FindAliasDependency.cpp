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
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <limits>

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
// Allocation records
//===----------------------------------------------------------------------===//

/// One memref allocation as seen by this pass after LocalStaging has assigned
/// physical addresses. `op` is the SSA producer (typically memref.alloc), and
/// `memref` is its result Value — kept so we can later find readers/writers
/// via getUsers() without re-walking from `op`.
struct Alloc {
  Operation *op;
  Value memref;
  int64_t tier;
  int64_t offset;
  int64_t size; // bytes
  int64_t time; // opTime[op]
};

/// Compute the byte size of a statically-shaped MemRefType.
/// Mirrors the calculation in MemRefLivenessAnalysis; if/when both grow more
/// users this should move into a shared header.
static int64_t bytesOf(MemRefType memrefType) {
  int64_t elementBytes =
      memrefType.getElementType().getIntOrFloatBitWidth() / 8;
  int64_t numElements = 1;
  for (int64_t dim : memrefType.getShape()) {
    if (dim != ShapedType::kDynamic)
      numElements *= dim;
  }
  return numElements * elementBytes;
}

/// Collect every memref-producing op in `funcOp` that carries both
/// `eaac.offset` and `eaac.tier`, returning the records sorted by
/// (tier, time) so same-tier allocations are contiguous in temporal order.
static llvm::SmallVector<Alloc>
collectAllocs(func::FuncOp funcOp,
              const llvm::DenseMap<Operation *, int64_t> &opTime) {
  llvm::SmallVector<Alloc> allocs;
  funcOp.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      auto memrefType = dyn_cast<MemRefType>(result.getType());
      if (!memrefType)
        continue;
      auto offsetAttr = op->getAttrOfType<IntegerAttr>("eaac.offset");
      auto tierAttr = op->getAttrOfType<IntegerAttr>("eaac.tier");
      if (!offsetAttr || !tierAttr)
        break; // not an EAAC-allocated memref; skip the whole op
      allocs.push_back({op, result, tierAttr.getInt(), offsetAttr.getInt(),
                        bytesOf(memrefType), opTime.lookup(op)});
      break; // one record per op even if it has multiple memref results
    }
  });
  llvm::sort(allocs, [](const Alloc &a, const Alloc &b) {
    return std::tie(a.tier, a.time) < std::tie(b.tier, b.time);
  });
  return allocs;
}

//===----------------------------------------------------------------------===//
// Async.execute lookups
//===----------------------------------------------------------------------===//

/// Walk parent ops to find the enclosing async.execute, if any.
static async::ExecuteOp enclosingExecute(Operation *op) {
  while (op && !isa<async::ExecuteOp>(op))
    op = op->getParentOp();
  return op ? cast<async::ExecuteOp>(op) : nullptr;
}

/// True iff `op` declares a value-bound Write effect on `memref`. Mirrors the
/// helper in FindAsyncDependency.cpp; if it grows a third user, lift to a
/// shared header.
static bool writesTo(Operation *op, Value memref) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    if (effect.getValue() == memref)
      return true;
  }
  return false;
}

/// async.execute with the largest op-time that has any user of `memref` in
/// its body. Used as the "last access" endpoint of the older buffer in an
/// alias pair. Returns nullptr if no such execute exists (e.g. memref was
/// allocated but never accessed inside an async.execute).
static async::ExecuteOp
lastAccessor(Value memref,
             const llvm::DenseMap<Operation *, int64_t> &opTime) {
  async::ExecuteOp best = nullptr;
  int64_t bestTime = -1;
  for (Operation *user : memref.getUsers()) {
    async::ExecuteOp exec = enclosingExecute(user);
    if (!exec)
      continue;
    int64_t t = opTime.lookup(exec);
    if (t > bestTime) {
      bestTime = t;
      best = exec;
    }
  }
  return best;
}

/// async.execute with the smallest op-time whose body contains a write
/// effect on `memref`. Used as the "first write" endpoint of the newer
/// buffer in an alias pair. Returns nullptr if nothing writes `memref`.
static async::ExecuteOp
firstWriter(Value memref,
            const llvm::DenseMap<Operation *, int64_t> &opTime) {
  async::ExecuteOp best = nullptr;
  int64_t bestTime = std::numeric_limits<int64_t>::max();
  for (Operation *user : memref.getUsers()) {
    if (!writesTo(user, memref))
      continue;
    async::ExecuteOp exec = enclosingExecute(user);
    if (!exec)
      continue;
    int64_t t = opTime.lookup(exec);
    if (t < bestTime) {
      bestTime = t;
      best = exec;
    }
  }
  return best;
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
    llvm::SmallVector<Alloc> allocs = collectAllocs(funcOp, opTime);

    LLVM_DEBUG({
      llvm::dbgs() << "[find-alias-dep] " << allocs.size()
                   << " EAAC-allocated memrefs in @"
                   << funcOp.getSymName() << "\n";
      for (const Alloc &a : allocs)
        llvm::dbgs() << "  tier=" << a.tier << " offset=" << a.offset
                     << " size=" << a.size << " time=" << a.time << "\n";
    });

    // Scan same-tier adjacent allocations for overlapping address ranges.
    // `allocs` is sorted by (tier, time), so the inner range starts after A
    // and terminates as soon as we leave A's tier.
    for (auto [i, a] : llvm::enumerate(allocs)) {
      const int64_t aHi = a.offset + a.size;
      for (const Alloc &b : llvm::ArrayRef(allocs).drop_front(i + 1)) {
        if (b.tier != a.tier)
          break;
        const int64_t bHi = b.offset + b.size;
        if (b.offset >= aHi || a.offset >= bHi)
          continue; // address ranges are disjoint

        // a is earlier than b in op-time (sort order). The anti-dep we want
        // is: firstWriter(b) must wait for lastAccessor(a).
        async::ExecuteOp reader = lastAccessor(a.memref, opTime);
        async::ExecuteOp writer = firstWriter(b.memref, opTime);
        if (!reader || !writer)
          continue; // nothing to chain to

        const bool covered = transitivelyOrdered(writer, reader, opTime);

        LLVM_DEBUG({
          llvm::dbgs() << "[alias] tier=" << a.tier
                       << " A.offset=" << a.offset
                       << " B.offset=" << b.offset
                       << " lastAccessor(A)@" << opTime.lookup(reader)
                       << " firstWriter(B)@" << opTime.lookup(writer)
                       << (covered ? " (already ordered)\n"
                                   : " NEEDS EDGE\n");
        });

        // TODO: if !covered, emit token edge writer -> reader; if the
        // distance exceeds the architectural sync window, fall back to a
        // fence. See FindAsyncDependency::rewriteWithDeps for the rebuild
        // pattern.
        (void)covered;
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createFindAliasDependencyPass() {
  return std::make_unique<FindAliasDependencyPass>();
}

} // namespace eaac
} // namespace mlir
