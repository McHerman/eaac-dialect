//===- AssignSemaphoreAddresses.cpp - Static semaphore address assignment --===//
//
// Assigns static hardware addresses to semaphore pairs using a linear scan
// allocator. Semaphores are self-sequencing (a producer/consumer cannot use
// a semaphore before it is initialized, and cannot reinitialize before done),
// so no spilling is needed under normal conditions.
//
// Each interval runs from its sem_alloc to its sem_dealloc. If we run out of
// addresses, we reuse the soonest-finishing one and mark it in the IR.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"
#include "eaac/StreamingChannelAnalysis.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "assign-semaphore-addresses"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_ASSIGNSEMAPHOREADDRESSES
#include "eaac/Passes.h.inc"

namespace {

/// A live interval for a semaphore value, spanning [sem_alloc, sem_dealloc).
struct SemInterval {
  Value semaphore;      // The SSA value from sem_alloc
  Operation *allocOp;   // The sem_alloc operation
  Operation *deallocOp; // The sem_dealloc operation
  int64_t start;        // Op number of sem_alloc
  int64_t end;          // Op number of sem_dealloc
  int64_t address;      // Assigned hardware address (-1 = unassigned)
  int64_t generation;   // Per-address generation tag (wraps mod num_generations)
  bool reused;          // True if this reuses an address from a live semaphore
  // Channel binding (from StreamingChannelAnalysis). When set, the semaphore
  // rotates through its channel's pre-reserved ring instead of going through
  // the linear-scan pool.
  llvm::StringRef channelName;
  int64_t channelDepth = 1;
  int64_t channelIdx = -1;
  bool onChannel() const { return !channelName.empty(); }

  SemInterval()
      : allocOp(nullptr), deallocOp(nullptr), start(0), end(0), address(-1),
        generation(0), reused(false) {}

  bool operator<(const SemInterval &other) const { return start < other.start; }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Look up an integer entry by `key` on the EAAC device entry of the module's
/// `dlti.target_system_spec`. The spec is mandatory.
static FailureOr<int64_t> getEaacIntegerEntry(ModuleOp module, StringRef key) {
  auto sysSpec = dyn_cast_or_null<TargetSystemSpecAttr>(
      module->getAttr(DLTIDialect::kTargetSystemDescAttrName));
  if (!sysSpec)
    return module.emitError("missing 'dlti.target_system_spec' on module");

  auto deviceId = StringAttr::get(module.getContext(), "EAAC");
  std::optional<TargetDeviceSpecInterface> deviceSpec =
      sysSpec.getDeviceSpecForDeviceID(deviceId);
  if (!deviceSpec)
    return module.emitError(
        "missing 'EAAC' device entry in 'dlti.target_system_spec'");

  for (DataLayoutEntryInterface entry : (*deviceSpec).getEntries()) {
    auto entryKey = dyn_cast<StringAttr>(entry.getKey());
    if (!entryKey || entryKey.getValue() != key)
      continue;
    auto i = dyn_cast<IntegerAttr>(entry.getValue());
    if (!i)
      return module.emitError("'") << key << "': expected integer, got "
                                   << entry.getValue();
    return i.getInt();
  }
  return module.emitError("'") << key << "' not found in EAAC device spec";
}

static FailureOr<int64_t> getEaacNumSemaphorePairs(ModuleOp module) {
  return getEaacIntegerEntry(module, "num_semaphore_pairs");
}

static FailureOr<int64_t> getEaacNumSemaphoreGenerations(ModuleOp module) {
  return getEaacIntegerEntry(module, "num_semaphore_generations");
}

class AssignSemaphoreAddressesPass
    : public impl::AssignSemaphoreAddressesBase<AssignSemaphoreAddressesPass> {
public:
  using AssignSemaphoreAddressesBase::AssignSemaphoreAddressesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<int64_t> numPairs = getEaacNumSemaphorePairs(module);
    if (failed(numPairs))
      return signalPassFailure();
    FailureOr<int64_t> numGenerations = getEaacNumSemaphoreGenerations(module);
    if (failed(numGenerations))
      return signalPassFailure();
    if (*numGenerations <= 0) {
      module.emitError("'num_semaphore_generations' must be > 0");
      return signalPassFailure();
    }

    auto &channelAnalysis = getAnalysis<StreamingChannelAnalysis>();

    auto result = module.walk([&](func::FuncOp funcOp) -> WalkResult {
      if (failed(processFunction(funcOp, *numPairs, *numGenerations,
                                 channelAnalysis)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }

private:
  /// Number all operations in a function sequentially.
  llvm::DenseMap<Operation *, int64_t> numberOperations(func::FuncOp funcOp) {
    llvm::DenseMap<Operation *, int64_t> opNumbers;
    int64_t counter = 0;
    funcOp.walk([&](Operation *op) { opNumbers[op] = counter++; });
    return opNumbers;
  }

  /// Build live intervals for all semaphore values.
  llvm::SmallVector<SemInterval>
  buildIntervals(func::FuncOp funcOp,
                 const llvm::DenseMap<Operation *, int64_t> &opNumbers,
                 const StreamingChannelAnalysis &channelAnalysis) {
    llvm::SmallVector<SemInterval> intervals;
    llvm::DenseMap<Value, size_t> semToIdx;


    // Collect all sem_alloc ops.
    /*
    funcOp.walk([&](SemAllocOp allocOp) {
      SemInterval interval;
      interval.semaphore = allocOp.getSemaphore();
      interval.allocOp = allocOp;
      interval.start = opNumbers.lookup(allocOp);
      interval.end = interval.start;

      if (auto ch = channelAnalysis.getAssignment(allocOp.getSemaphore())) {
        interval.channelName = ch->name;
        interval.channelDepth = ch->depth;
        interval.channelIdx = ch->index;
      }

      semToIdx[allocOp.getSemaphore()] = intervals.size();
      intervals.push_back(interval);
    });
    */

    // Per-semaphore extended end times produced by chain dependencies.
    // Populated below as we walk sem_allocs; consumed by the dealloc loop.
    llvm::DenseMap<Value, int64_t> chainedSemIntervalEnd;

    llvm::SmallVector<eaac::SemAllocOp> semAllocOps;
    funcOp.walk([&](eaac::SemAllocOp op) { semAllocOps.push_back(op); });

    for (eaac::SemAllocOp semAllocOp : semAllocOps) {
      SemInterval interval;
      interval.semaphore = semAllocOp.getSemaphore();
      interval.allocOp = semAllocOp;
      interval.start = opNumbers.lookup(semAllocOp);
      interval.end = interval.start;

      /*
      if (auto ch = channelAnalysis.getAssignment(semAllocOp.getSemaphore())) {
        interval.channelName = ch->name;
        interval.channelDepth = ch->depth;
        interval.channelIdx = ch->index;
      }
      */

      semToIdx[semAllocOp.getSemaphore()] = intervals.size();

      const int64_t allocTime = opNumbers.lookup(semAllocOp);
      for (Value chainedSem : semAllocOp.getChainsFrom()) {
        if (auto predAllocOp = chainedSem.getDefiningOp<eaac::SemAllocOp>()) {
          // Bump the predecessor's required end to at least this alloc's
          // start (its address must remain reserved until this alloc fires).
          auto [it, inserted] = chainedSemIntervalEnd.try_emplace(
              predAllocOp.getSemaphore(), allocTime + 1);
          if (!inserted)
            it->second = std::max(it->second, allocTime + 1);
          LLVM_DEBUG(llvm::dbgs()
                     << "[chain] extend pred sem live-range to "
                     << it->second
                     << " (chained from alloc @" << interval.start << ")\n");
        }
      }

      intervals.push_back(interval);
    }



    // Find sem_dealloc ops to set interval ends.
    funcOp.walk([&](SemDeallocOp deallocOp) {
      Value sem = deallocOp.getSemaphore();
      auto it = semToIdx.find(sem);
      if (it == semToIdx.end()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[warn] sem_dealloc has no matching sem_alloc interval\n");
        return;
      }

      intervals[it->second].deallocOp = deallocOp;
      const int64_t natural = opNumbers.lookup(deallocOp);

      auto extendedIt = chainedSemIntervalEnd.find(sem);
      if (extendedIt != chainedSemIntervalEnd.end() &&
          extendedIt->second > natural) {
        intervals[it->second].end = extendedIt->second;
        LLVM_DEBUG(llvm::dbgs()
                   << "[chain] sem @" << intervals[it->second].start
                   << " end extended to " << intervals[it->second].end
                   << " past natural sem_dealloc (chain target)\n");
      } else {
        intervals[it->second].end = natural;
      }
    });

    // Sort by start position.
    std::sort(intervals.begin(), intervals.end());

    LLVM_DEBUG({
      llvm::dbgs() << "Semaphore intervals (" << intervals.size() << "):\n";
      for (const auto &iv : intervals) {
        llvm::dbgs() << "  sem @" << iv.start << "-" << iv.end;
        /*
        if (iv.onChannel())
          llvm::dbgs() << " [" << iv.channelName << " idx=" << iv.channelIdx
                       << "/" << iv.channelDepth << "]";
        */
        llvm::dbgs() << "\n";
      }
    });

    return intervals;
  }

  /// Run the linear scan allocator. Returns failure if an eviction is unsafe.
  ///
  /// Channel-bound intervals (identified by StreamingChannelAnalysis) bypass
  /// the linear scan entirely and rotate through a per-channel ring carved
  /// out at the top of the address space. The remaining intervals use the
  /// linear scan against `[0, scanCap)`. The pools are disjoint so the linear
  /// scan never observes ring-occupied addresses.
  LogicalResult allocate(llvm::SmallVector<SemInterval> &intervals,
                         int64_t numPairs, int64_t numGenerations) {
    // Per-hardware-address generation counter. Each time an address is
    // assigned to a sem_alloc, that address's counter advances; the resulting
    // tag is written onto the interval and wraps modulo numGenerations.
    llvm::DenseMap<int64_t, int64_t> nextGeneration;
    auto takeGeneration = [&](int64_t addr) {
      int64_t &counter = nextGeneration[addr];
      int64_t gen = counter % numGenerations;
      counter++;
      return gen;
    };
    /*
    // Discover the channels actually used in this function (subset of what
    // the analysis surfaced module-wide), then reserve a ring per channel at
    // the top of the address space. Sort by name for deterministic layout.
    llvm::SmallVector<ChannelSpec> usedChannels;
    {
      llvm::StringMap<int64_t> depthByName;
      for (const auto &iv : intervals)
        if (iv.onChannel())
          depthByName.try_emplace(iv.channelName, iv.channelDepth);
      for (auto &kv : depthByName)
        usedChannels.push_back({kv.first(), kv.second});
      llvm::sort(usedChannels,
                 [](const ChannelSpec &a, const ChannelSpec &b) {
                   return a.name < b.name;
                 });
    }

    llvm::StringMap<int64_t> channelBase;
    int64_t cursor = numPairs;
    for (const auto &ch : usedChannels) {
      cursor -= ch.depth;
      channelBase[ch.name] = cursor;
    }
    const int64_t scanCap = cursor;

    if (scanCap < 0) {
      intervals.front().allocOp->emitError("num_semaphore_pairs (")
          << numPairs
          << ") too small to reserve channel rings totalling " << (numPairs - scanCap)
          << " addresses";
      return failure();
    }
    */
    const int64_t scanCap = numPairs;

    // Pre-assign channel-bound intervals. No liveness/eviction check: the
    // architectural drain depth K guarantees the previous slot occupant is
    // done by the time we wrap around.
    /*
    for (auto &iv : intervals) {
      if (!iv.onChannel())
        continue;
      const int64_t base = channelBase[iv.channelName];
      iv.address = base + (iv.channelIdx % iv.channelDepth);
      iv.reused = iv.channelIdx >= iv.channelDepth;
      iv.generation = takeGeneration(iv.address);
      LLVM_DEBUG(llvm::dbgs()
                 << "  channel-ring(" << iv.channelName
                 << "): addr=" << iv.address << " gen=" << iv.generation
                 << " for sem @" << iv.start << "-" << iv.end
                 << " (idx=" << iv.channelIdx << ")\n");
    }
    */

    llvm::SmallVector<SemInterval *> active;
    int64_t hwm = 0;
    llvm::SmallVector<int64_t> lastEnd(scanCap, -1);

    for (auto &cur : intervals) {
      /*
      if (cur.onChannel())
        continue; // already placed in a ring
      */

      // Expire finished intervals.
      llvm::SmallVector<SemInterval *> stillActive;
      for (auto *a : active) {
        if (a->end <= cur.start) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  expired addr=" << a->address << " @" << a->end
                     << "\n");
        } else {
          stillActive.push_back(a);
        }
      }
      active = std::move(stillActive);

      // Collect free addresses (within the scan range only).
      llvm::SmallVector<bool> used(scanCap, false);
      for (auto *a : active)
        used[a->address] = true;

      // Prefer a never-issued address; only recycle once hwm hits scanCap.
      int64_t freeAddr = -1;
      if (hwm < scanCap) {
        freeAddr = hwm++;
      } else {
        // Pick the free address whose previous user finished earliest
        // (least-recently-used). Ties broken by lowest index for determinism.
        int64_t bestEnd = INT64_MAX;
        for (int64_t i = 0; i < scanCap; ++i) {
          if (!used[i] && lastEnd[i] < bestEnd) {
            bestEnd = lastEnd[i];
            freeAddr = i;
          }
        }
      }

      if (freeAddr >= 0) {
        cur.address = freeAddr;
        cur.generation = takeGeneration(freeAddr);
        lastEnd[freeAddr] = cur.end;
        active.push_back(&cur);
        LLVM_DEBUG(llvm::dbgs() << "  assigned addr=" << freeAddr
                                << " gen=" << cur.generation << " to sem @"
                                << cur.start << "-" << cur.end << "\n");
        continue;
      }

      // --- No free address: evict the soonest-finishing interval ---

      SemInterval *victim = nullptr;
      int64_t earliestEnd = INT64_MAX;
      for (auto *a : active) {
        if (a->end < earliestEnd) {
          earliestEnd = a->end;
          victim = a;
        }
      }

      assert(victim && "active list must be non-empty if no free address");

      // Safety check: the victim's sem_dealloc (which the lowering pass places
      // after all consumers complete) must precede cur's sem_alloc. Otherwise
      // reinitializing the address would clobber state the victim's consumers
      // still need to read.
      if (victim->end > cur.start) {
        cur.allocOp->emitError(
            "out of semaphore addresses and cannot safely reuse: "
            "candidate semaphore (addr=")
            << victim->address << ") is still live (sem_dealloc at op "
            << victim->end << " is after this sem_alloc at op " << cur.start
            << ")";
        return failure();
      }

      cur.address = victim->address;
      cur.generation = takeGeneration(cur.address);
      cur.reused = true;

      LLVM_DEBUG(llvm::dbgs()
                 << "  REUSE: addr=" << cur.address
                 << " gen=" << cur.generation << " from sem @"
                 << victim->start << "-" << victim->end << " for sem @"
                 << cur.start << "-" << cur.end << "\n");

      // Replace victim with current in active list.
      for (auto &a : active) {
        if (a == victim) {
          a = &cur;
          break;
        }
      }
    }
    return success();
  }

  /// Drop sem_require ops whose producer is on the same channel at queue
  /// distance >= K — the FU's drain pipeline already serializes them. Then
  /// sweep any sem_alloc that's been left without sem_require users (along
  /// with its matching sem_acquire / sem_dealloc). Runs before buildIntervals
  /// so the allocator never sees the elided semaphores.
  void elideRedundantChannelSemaphores(
      func::FuncOp funcOp, const StreamingChannelAnalysis &analysis) {
    llvm::SmallVector<SemRequireOp> reqs;
    funcOp.walk([&](SemRequireOp r) { reqs.push_back(r); });
    for (SemRequireOp req : reqs) {
      auto prod = analysis.getAssignment(req.getSemaphore());
      auto cons = analysis.getAssignment(req->getParentOfType<ExecuteOp>());
      if (!prod || !cons || prod->name != cons->name ||
          cons->index - prod->index < prod->depth)
        continue;
      LLVM_DEBUG(llvm::dbgs() << "  elide sem_require: " << prod->name << "["
                              << prod->index << "] -> [" << cons->index
                              << "]\n");
      req.getResult().replaceAllUsesWith(req.getMemref());
      req.erase();
    }

    // Sweep orphaned sem_allocs (and their sem_acquire / sem_dealloc).
    // A sem is sweepable iff every non-acquire/dealloc user is another
    // SemAllocOp carrying it in `chains_from`. SemRequireOp (or anything
    // else unexpected) means the semaphore is still load-bearing and we
    // skip it. The orphan sweep is correctness-required: the assembler
    // links every generation N to N+1 on the same address, so a surviving
    // orphan whose signal never fires would deadlock the next generation.
    //
    // Dropping the chain edge when we sweep the chain target is safe
    // because the only orphans we sweep are sems whose consumer waits were
    // elided by `elideRedundantChannelSemaphores`, which already proved
    // those producers/consumers are queue-serialized at the FU level. The
    // chain existed as anti-aliasing protection — that protection is
    // redundant once we've established queue order.
    llvm::SmallVector<SemAllocOp> allocs;
    funcOp.walk([&](SemAllocOp a) { allocs.push_back(a); });
    for (SemAllocOp alloc : allocs) {
      Value sem = alloc.getSemaphore();
      bool sweepable = true;
      llvm::SmallVector<SemAllocOp> chainUsers;
      for (Operation *u : sem.getUsers()) {
        if (isa<SemAcquireOp, SemDeallocOp>(u))
          continue;
        if (auto chainUser = dyn_cast<SemAllocOp>(u)) {
          chainUsers.push_back(chainUser);
          continue;
        }
        sweepable = false;
        break;
      }
      if (!sweepable)
        continue;

      // Drop this sem from each chain user's chains_from operand list.
      // Iterate indices high-to-low so the erases don't shift positions
      // we still need to examine. With only one variadic on SemAllocOp
      // there's no operand-segment-sizes attribute to update.
      for (SemAllocOp chainUser : chainUsers) {
        Operation *op = chainUser.getOperation();
        for (int i = op->getNumOperands() - 1; i >= 0; --i) {
          if (op->getOperand(i) == sem)
            op->eraseOperand(i);
        }
      }

      // Snapshot remaining users (acquires + deallocs) and erase.
      for (Operation *u : llvm::SmallVector<Operation *>(sem.getUsers())) {
        if (auto acq = dyn_cast<SemAcquireOp>(u)) {
          acq.getResult().replaceAllUsesWith(acq.getMemref());
          acq.erase();
        } else {
          u->erase(); // sem_dealloc
        }
      }
      alloc.erase();
    }
  }

  /// Annotate sem_alloc ops with assigned addresses.
  void annotateIR(const llvm::SmallVector<SemInterval> &intervals) {
    for (const auto &iv : intervals) {
      auto *op = iv.allocOp;
      auto ctx = op->getContext();

      op->setAttr("eaac.sem_addr",
                  IntegerAttr::get(IndexType::get(ctx), iv.address));
      op->setAttr("eaac.sem_gen",
                  IntegerAttr::get(IndexType::get(ctx), iv.generation));

      /*
      if (iv.onChannel()) {
        // Ring reuse is architecturally safe (drain pipeline guarantees the
        // previous slot occupant is done), so no warning here.
        op->setAttr("eaac.sem_ring", StringAttr::get(ctx, iv.channelName));
      } else
      */
      if (iv.reused) {
        op->setAttr("eaac.sem_reused", UnitAttr::get(ctx));
        op->emitWarning("semaphore address reused due to register pressure — "
                        "correctness depends on self-sequencing guarantee");
      }
    }
  }

  LogicalResult
  processFunction(func::FuncOp funcOp, int64_t numPairs,
                  int64_t numGenerations,
                  const StreamingChannelAnalysis &channelAnalysis) {
    //elideRedundantChannelSemaphores(funcOp, channelAnalysis);
    auto opNumbers = numberOperations(funcOp);
    auto intervals = buildIntervals(funcOp, opNumbers, channelAnalysis);

    if (intervals.empty())
      return success();

    if (failed(allocate(intervals, numPairs, numGenerations)))
      return failure();

    annotateIR(intervals);
    return success();
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createAssignSemaphoreAddressesPass() {
  return std::make_unique<AssignSemaphoreAddressesPass>();
}

} // namespace eaac
} // namespace mlir
