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
  bool reused;          // True if this reuses an address from a live semaphore

  SemInterval()
      : allocOp(nullptr), deallocOp(nullptr), start(0), end(0), address(-1),
        reused(false) {}

  bool operator<(const SemInterval &other) const { return start < other.start; }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class AssignSemaphoreAddressesPass
    : public impl::AssignSemaphoreAddressesBase<AssignSemaphoreAddressesPass> {
public:
  using AssignSemaphoreAddressesBase::AssignSemaphoreAddressesBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto result = module.walk([&](func::FuncOp funcOp) -> WalkResult {
      if (failed(processFunction(funcOp)))
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
                 const llvm::DenseMap<Operation *, int64_t> &opNumbers) {
    llvm::SmallVector<SemInterval> intervals;
    llvm::DenseMap<Value, size_t> semToIdx;

    // Collect all sem_alloc ops.
    funcOp.walk([&](SemAllocOp allocOp) {
      SemInterval interval;
      interval.semaphore = allocOp.getSemaphore();
      interval.allocOp = allocOp;
      interval.start = opNumbers.lookup(allocOp);
      interval.end = interval.start;
      semToIdx[allocOp.getSemaphore()] = intervals.size();
      intervals.push_back(interval);
    });

    // Find sem_dealloc ops to set interval ends.
    funcOp.walk([&](SemDeallocOp deallocOp) {
      Value sem = deallocOp.getSemaphore();
      auto it = semToIdx.find(sem);
      if (it != semToIdx.end()) {
        intervals[it->second].deallocOp = deallocOp;
        intervals[it->second].end = opNumbers.lookup(deallocOp);
      }
    });

    // Sort by start position.
    std::sort(intervals.begin(), intervals.end());

    LLVM_DEBUG({
      llvm::dbgs() << "Semaphore intervals (" << intervals.size() << "):\n";
      for (const auto &iv : intervals)
        llvm::dbgs() << "  sem @" << iv.start << "-" << iv.end << "\n";
    });

    return intervals;
  }

  /// Run the linear scan allocator. Returns failure if an eviction is unsafe.
  LogicalResult allocate(llvm::SmallVector<SemInterval> &intervals,
                         int64_t numPairs) {
    llvm::SmallVector<SemInterval *> active;
    // High-water mark across the whole allocation: addresses [0, hwm) have
    // been issued at least once. We hand out a never-issued address before
    // recycling a freed one — pre-loaded command queues on infrequently-used
    // FUs (e.g. the store unit) can otherwise see signaling on a recycled
    // address and falsely trigger.
    int64_t hwm = 0;

    for (auto &cur : intervals) {
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

      // Collect free addresses.
      llvm::SmallVector<bool> used(numPairs, false);
      for (auto *a : active)
        used[a->address] = true;

      // Prefer a never-issued address; only recycle once hwm hits numPairs.
      int64_t freeAddr = -1;
      if (hwm < numPairs) {
        freeAddr = hwm++;
      } else {
        for (int64_t i = 0; i < numPairs; ++i) {
          if (!used[i]) {
            freeAddr = i;
            break;
          }
        }
      }

      if (freeAddr >= 0) {
        cur.address = freeAddr;
        active.push_back(&cur);
        LLVM_DEBUG(llvm::dbgs() << "  assigned addr=" << freeAddr << " to sem @"
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
      cur.reused = true;

      LLVM_DEBUG(llvm::dbgs()
                 << "  REUSE: addr=" << cur.address << " from sem @"
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

  /// Annotate sem_alloc ops with assigned addresses.
  void annotateIR(const llvm::SmallVector<SemInterval> &intervals) {
    for (const auto &iv : intervals) {
      auto *op = iv.allocOp;
      auto ctx = op->getContext();

      op->setAttr("eaac.sem_addr",
                  IntegerAttr::get(IndexType::get(ctx), iv.address));

      if (iv.reused) {
        op->setAttr("eaac.sem_reused", UnitAttr::get(ctx));
        op->emitWarning("semaphore address reused due to register pressure — "
                        "correctness depends on self-sequencing guarantee");
      }
    }
  }

  LogicalResult processFunction(func::FuncOp funcOp) {
    auto opNumbers = numberOperations(funcOp);
    auto intervals = buildIntervals(funcOp, opNumbers);

    if (intervals.empty())
      return success();

    if (failed(allocate(intervals, numSemaphorePairs)))
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
