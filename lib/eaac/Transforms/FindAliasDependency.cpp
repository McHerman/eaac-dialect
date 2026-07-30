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
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <limits>
#include <optional>

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

/// Find the eaac.require inside `exec` whose memref operand is `memref`.
/// Returns a null op if no such require exists.
static RequireOp findRequireForMemref(async::ExecuteOp exec, Value memref) {
  RequireOp found = nullptr;
  exec.getBody()->walk([&](RequireOp r) {
    if (r.getMemref() == memref) {
      found = r;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
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
// DLTI lookup
//===----------------------------------------------------------------------===//

/// Read `alias_check_length` from the EAAC device entry of the module's
/// `dlti.target_system_spec`. Returns nullopt if either the spec or the key
/// is absent — the pass then runs with no time-window filter, i.e. checks
/// every aliasing pair regardless of their op-time distance.
static std::optional<int64_t> getEaacAliasCheckLength(ModuleOp module) {
  auto sysSpec = dyn_cast_or_null<TargetSystemSpecAttr>(
      module->getAttr(DLTIDialect::kTargetSystemDescAttrName));
  if (!sysSpec)
    return std::nullopt;
  auto deviceId = StringAttr::get(module.getContext(), "EAAC");
  std::optional<TargetDeviceSpecInterface> deviceSpec =
      sysSpec.getDeviceSpecForDeviceID(deviceId);
  if (!deviceSpec)
    return std::nullopt;
  for (DataLayoutEntryInterface entry : (*deviceSpec).getEntries()) {
    auto key = dyn_cast<StringAttr>(entry.getKey());
    if (!key || key.getValue() != "alias_check_length")
      continue;
    auto intAttr = dyn_cast<IntegerAttr>(entry.getValue());
    if (!intAttr) {
      module.emitWarning() << "'alias_check_length': expected i64, got "
                           << entry.getValue() << "; ignoring";
      return std::nullopt;
    }
    return intAttr.getInt();
  }
  return std::nullopt;
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
    std::optional<int64_t> aliasCheckLength = getEaacAliasCheckLength(module);
    LLVM_DEBUG({
      if (aliasCheckLength)
        llvm::dbgs() << "[find-alias-dep] alias_check_length="
                     << *aliasCheckLength << "\n";
      else
        llvm::dbgs() << "[find-alias-dep] alias_check_length unset; "
                        "checking every pair\n";
    });
    module.walk([&](func::FuncOp funcOp) {
      auto pairs = collectAliasPairs(funcOp, aliasCheckLength);
      LLVM_DEBUG({
        llvm::dbgs() << "[find-alias-dep] " << pairs.size()
                     << " alias pairs in @" << funcOp.getSymName() << "\n";
        for (auto [reader, writer, memref] : pairs)
          llvm::dbgs() << "  reader=" << reader << " writer=" << writer
                       << "\n";
      });
      for (auto [readerExec, writerExec, readerMemref] : pairs) {
        // readerExec : async.execute wrapping the last access of the older
        //              buffer (the predecessor on the aliased address).
        // writerExec : async.execute wrapping the first write of the newer
        //              buffer (the successor that overwrites the address).

        RequireOp readerRequire = findRequireForMemref(readerExec, readerMemref);
        if (!readerRequire) {
          readerExec.emitWarning(
              "eaac.chain: no eaac.require found for aliased memref in "
              "reader exec; dropping alias edge");
          continue;
        }

        OpBuilder builder(writerExec.getBody(), writerExec.getBody()->begin());
        ChainOp::create(builder, writerExec.getLoc(), readerRequire.getToken(),
                        /*is_broadcast=*/false);
      }
    });
  }

private:
  /// A reader/writer pair on aliased addresses. `reader` is the async.execute
  /// whose body holds the last access of the older buffer; `writer` is the
  /// async.execute whose body holds the first write to the newer buffer that
  /// will physically overwrite the older one. `readerMemref` is the aliased
  /// memref value accessed in `reader`, needed to locate its eaac.require.
  struct AliasPair {
    async::ExecuteOp reader;
    async::ExecuteOp writer;
    Value readerMemref;
  };

  /// Walk `funcOp` and return every pair of async.executes whose memrefs
  /// share an {tier, [offset, offset+size)} range and are not already
  /// transitively ordered by the existing async.token graph. The list is
  /// filtered by the DLTI `alias_check_length` window when provided.
  llvm::SmallVector<AliasPair>
  collectAliasPairs(func::FuncOp funcOp,
                    std::optional<int64_t> aliasCheckLength) {
    llvm::SmallVector<AliasPair> pairs;
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
    // and terminates as soon as we leave A's tier — or, if a time-window
    // filter is set, as soon as B's op-time pulls beyond the window.
    for (auto [i, a] : llvm::enumerate(allocs)) {
      const int64_t aHi = a.offset + a.size;
      for (const Alloc &b : llvm::ArrayRef(allocs).drop_front(i + 1)) {
        if (b.tier != a.tier)
          break;
        if (aliasCheckLength && (b.time - a.time) > *aliasCheckLength)
          break; // sorted by time within tier; later b's exceed the window
        const int64_t bHi = b.offset + b.size;
        if (b.offset >= aHi || a.offset >= bHi)
          continue; // address ranges are disjoint

        // a is earlier than b in op-time (sort order). The anti-dep we want
        // is: firstWriter(b) must wait for lastAccessor(a).
        async::ExecuteOp reader = lastAccessor(a.memref, opTime);
        async::ExecuteOp writer = firstWriter(b.memref, opTime);
        if (!reader || !writer)
          continue; // nothing to chain to

        if (transitivelyOrdered(writer, reader, opTime))
          continue; // already covered by the existing token graph

        pairs.push_back({reader, writer, a.memref});
      }
    }

    // Ties chaining to latest writer instead of all writers in scope
    llvm::DenseMap<Operation *, size_t> bestForWriter;
    llvm::SmallVector<AliasPair> deduped;
    for (AliasPair p : pairs) {
      auto [it, inserted] = bestForWriter.try_emplace(p.writer.getOperation(), deduped.size());
      if (inserted) {
        deduped.push_back(p);
        continue;
      }
      AliasPair &existing = deduped[it->second];
      if (opTime.lookup(p.reader) > opTime.lookup(existing.reader))
        existing = p;
    }
    return deduped;
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createFindAliasDependencyPass() {
  return std::make_unique<FindAliasDependencyPass>();
}

} // namespace eaac
} // namespace mlir
