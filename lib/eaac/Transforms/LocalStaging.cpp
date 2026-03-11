//===- LocalStaging.cpp - Insert local SRAM staging for compute ops -===//
//
// Pass to insert memref.copy operations to stage data into local SRAM
// before compute operations and copy results back afterward.
//
//===----------------------------------------------------------------------===//

#include "eaac/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LOCALSTAGING
#include "eaac/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Buffer allocation types (equivalent to minimalloc Python types)
//===----------------------------------------------------------------------===//

/// Represents a time interval [lower, upper).
struct Interval {
  int64_t lower;
  int64_t upper;

  Interval() : lower(0), upper(0) {}
  Interval(int64_t lower, int64_t upper) : lower(lower), upper(upper) {}
};

/// Represents a memory buffer with its lifetime and allocation info.
struct Buffer {
  std::string id;
  Interval lifespan;
  int64_t size;
  int64_t alignment;
  int64_t offset; // Assigned offset after allocation

  Buffer() : size(0), alignment(1), offset(-1) {}
  Buffer(std::string id, Interval lifespan, int64_t size, int64_t alignment = 1)
      : id(std::move(id)), lifespan(lifespan), size(size), alignment(alignment),
        offset(-1) {}
};

//===----------------------------------------------------------------------===//
// Live interval and memory tier types
//===----------------------------------------------------------------------===//

/// Stores information about a single use of a memref.
struct MemRefUse {
  Operation *op;
  int64_t time;
};

/// A live range with use positions (equivalent to Python LiveInterval).
struct LiveInterval {
  std::string id;
  int64_t size;
  int64_t start;
  int64_t end;
  llvm::SmallVector<int64_t> uses;
  int64_t offset;
  std::optional<int64_t> reloadFromTier;
  std::optional<int64_t> reloadFromOffset;

  // Optional reference to the original allocation op
  memref::AllocOp allocOp;

  LiveInterval()
      : size(0), start(0), end(0), offset(-1), reloadFromTier(std::nullopt),
        reloadFromOffset(std::nullopt), allocOp(nullptr) {}

  LiveInterval(std::string id, int64_t size, int64_t start, int64_t end,
               llvm::SmallVector<int64_t> uses = {}, int64_t offset = -1)
      : id(std::move(id)), size(size), start(start), end(end),
        uses(std::move(uses)), offset(offset), reloadFromTier(std::nullopt),
        reloadFromOffset(std::nullopt), allocOp(nullptr) {}

  /// Comparison operator for sorting by start position.
  bool operator<(const LiveInterval &other) const { return start < other.start; }
};

/// Represents a memory tier with its capacity and state.
class MemoryTier {
public:
  std::string name;
  int64_t capacity;
  int64_t level;

  llvm::SmallVector<Buffer> buffers;
  llvm::SmallVector<LiveInterval> active;
  llvm::SmallVector<LiveInterval> handled;
  llvm::SmallVector<LiveInterval> unhandled;

  MemoryTier() : capacity(0), level(0) {}
  MemoryTier(std::string name, int64_t capacity, int64_t level)
      : name(std::move(name)), capacity(capacity), level(level) {}

  /// Remove and return a buffer by its id. Returns nullptr if not found.
  std::optional<Buffer> removeBuffer(llvm::StringRef bufferId) {
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      if (it->id == bufferId) {
        Buffer removed = std::move(*it);
        buffers.erase(it);
        return removed;
      }
    }
    return std::nullopt;
  }

  /// Find a buffer by its id. Returns nullptr if not found.
  Buffer *getBuffer(llvm::StringRef bufferId) {
    for (auto &buf : buffers) {
      if (buf.id == bufferId) {
        return &buf;
      }
    }
    return nullptr;
  }

  /// Remove and return an active interval by its id.
  std::optional<LiveInterval> removeActive(llvm::StringRef intervalId) {
    for (auto it = active.begin(); it != active.end(); ++it) {
      if (it->id == intervalId) {
        LiveInterval removed = std::move(*it);
        active.erase(it);
        return removed;
      }
    }
    return std::nullopt;
  }

  /// Add an interval to the unhandled list, maintaining sorted order by start.
  void addUnhandled(LiveInterval interval) {
    auto insertPos = std::lower_bound(unhandled.begin(), unhandled.end(),
                                       interval);
    unhandled.insert(insertPos, std::move(interval));
  }
};

/// A memory allocation problem instance.
struct AllocationProblem {
  int64_t numRegs;
  llvm::SmallVector<LiveInterval> intervals;

  AllocationProblem() : numRegs(0) {}
  explicit AllocationProblem(int64_t numRegs) : numRegs(numRegs) {}

  /// Add an interval with its liveness range and use positions.
  AllocationProblem &add(std::string id, int64_t size, int64_t start,
                         int64_t end, llvm::SmallVector<int64_t> uses = {},
                         int64_t offset = -1) {
    intervals.emplace_back(std::move(id), size, start, end, std::move(uses),
                           offset);
    return *this;
  }

  /// Return intervals sorted by start position.
  llvm::SmallVector<LiveInterval> sortedIntervals() const {
    llvm::SmallVector<LiveInterval> sorted = intervals;
    std::sort(sorted.begin(), sorted.end());
    return sorted;
  }
};

/// Result of computing a spill operation.
struct SpillResult {
  std::optional<LiveInterval> reloadInterval;
  std::optional<Buffer> spillBuffer;
  std::optional<Buffer> prefixBuffer;
  std::optional<LiveInterval> prefixInterval;

  SpillResult() = default;
  SpillResult(std::optional<LiveInterval> reloadInterval,
              std::optional<Buffer> spillBuffer,
              std::optional<Buffer> prefixBuffer,
              std::optional<LiveInterval> prefixInterval)
      : reloadInterval(std::move(reloadInterval)),
        spillBuffer(std::move(spillBuffer)),
        prefixBuffer(std::move(prefixBuffer)),
        prefixInterval(std::move(prefixInterval)) {}
};






class LocalStagingPass
    : public impl::LocalStagingBase<LocalStagingPass> {
public:
  using LocalStagingBase::LocalStagingBase;

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
      llvm::SmallVector<LiveInterval> lifetimeInfos;

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

        // Compute buffer size (simplified - assumes 1D for now)
        int64_t size = 1;
        auto memrefType = llvm::cast<MemRefType>(memref.getType());
        for (int64_t dim : memrefType.getShape()) {
          if (dim != ShapedType::kDynamic)
            size *= dim;
        }

        // Create and store the LiveInterval
        LiveInterval interval(id, size, startTime, endTime, std::move(useTimes));
        interval.allocOp = allocOp;
        lifetimeInfos.push_back(std::move(interval));

        // Print lifetime and usage information
        llvm::errs() << "MemRef allocation at time " << startTime << ":\n";
        llvm::errs() << "  Lifetime: [" << startTime << ", " << endTime << "]\n";
        llvm::errs() << "  Size: " << size << "\n";
        llvm::errs() << "  Uses (" << memRefUses.size() << "):\n";
        for (const auto &use : memRefUses) {
          llvm::errs() << "    t=" << use.time << ": " 
                       << use.op->getName().getStringRef() << "\n";
        }
        llvm::errs() << "\n";
      });
    });

    // TODO: Implement local SRAM staging
    // This pass should:
    // 1. Walk through linalg ops
    // 2. For each memref operand not in local SRAM (memory_space=3):
    //    - Allocate a local buffer
    //    - Insert memref.copy to stage data in
    // 3. For output operands:
    //    - Insert memref.copy to write results back
    // 4. Rewrite compute op to use local buffers
    // 5. Deallocate local buffers
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLocalStagingPass() {
  return std::make_unique<LocalStagingPass>();
}

} // namespace eaac
} // namespace mlir
