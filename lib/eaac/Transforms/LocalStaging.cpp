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
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include "minimalloc.h"
#include "solver.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LOCALSTAGING
#include "eaac/Passes.h.inc"

namespace {

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

    /*
    for (auto it = buffers.begin(); it != buffers.end(); ++it) {
      if (it->id == bufferId) {
        Buffer removed = std::move(*it);
        buffers.erase(it);
        return removed;
      }
    }
    */

    for (auto [i, buf] : llvm::enumerate(buffers)) {
      if (buf.id == bufferId) {
        Buffer removed = std::move(buf);
        buffers.erase(buffers.begin() + i);
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
    /*
    for (auto it = active.begin(); it != active.end(); ++it) {
      if (it->id == intervalId) {
        LiveInterval removed = std::move(*it);
        active.erase(it);
        return removed;
      }
    }
    */

    for (auto [i, interval] : llvm::enumerate(active)) {
      if (interval.id == intervalId) {
        LiveInterval removed = std::move(interval);
        active.erase(active.begin() + i);
        return removed;
      }
    }

    return std::nullopt;
  }


  /// Add an interval to the unhandled list, maintaining sorted order by start.
  void addUnhandled(LiveInterval interval) {
    auto pos = llvm::lower_bound(unhandled, interval);
    unhandled.insert(pos, std::move(interval));
  }
};

/// A memory allocation problem instance.
struct AllocationProblem {
  int64_t numRegs;
  llvm::SmallVector<LiveInterval> intervals;

  AllocationProblem() : numRegs(0) {}
  explicit AllocationProblem(int64_t numRegs) : numRegs(numRegs) {}

  /// Add an interval with its liveness range and use positions.
  AllocationProblem &add(LiveInterval liveinterval) {
    intervals.emplace_back(liveinterval);
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

//===----------------------------------------------------------------------===//
// Helper functions for tiered memory allocation
//===----------------------------------------------------------------------===//

/// Get LiveIntervals from active that appear in the subset (IIS indices).
llvm::SmallVector<LiveInterval *>
getIntervalsFromSubset(const std::vector<long> &subset,
                       llvm::SmallVector<Buffer> &buffers,
                       llvm::SmallVector<LiveInterval> &active) {
  llvm::SmallVector<LiveInterval *> result;
  if (subset.empty())
    return result;

  // Collect IDs of buffers in the subset
  llvm::StringSet<> conflictingIds;
  for (int idx : subset) {
    if (idx >= 0 && static_cast<size_t>(idx) < buffers.size()) {
      conflictingIds.insert(buffers[idx].id);
    }
  }

  // Find active intervals with those IDs
  for (auto &iv : active) {
    if (conflictingIds.contains(iv.id)) {
      result.push_back(&iv);
    }
  }
  return result;
}

/// Try to allocate a buffer in a tier. Returns solution if successful.
std::optional<minimalloc::Solution>
tryAllocate(MemoryTier &tier, const Buffer &buffer, minimalloc::Solver &solver) {
  llvm::errs() << "[tryAllocate] " << tier.name << ": trying buffer id="
               << buffer.id << " size=" << buffer.size << " lifespan=["
               << buffer.lifespan.lower << ", " << buffer.lifespan.upper
               << ") capacity=" << tier.capacity << "\n";
  llvm::errs() << "[tryAllocate] " << tier.name << ": existing buffers ("
               << tier.buffers.size() << "): ";
  for (const auto &buf : tier.buffers) {
    llvm::errs() << "{id=" << buf.id << " [" << buf.lifespan.lower << ","
                 << buf.lifespan.upper << ") sz=" << buf.size << "} ";
  }
  llvm::errs() << "\n";

  // Add buffer to tier
  tier.buffers.push_back(buffer);

  // Create minimalloc problem
  minimalloc::Problem problem;
  problem.capacity = tier.capacity;
  for (const auto &buf : tier.buffers) {
    problem.buffers.push_back(minimalloc::Buffer{
        .id = buf.id,
        .lifespan = {buf.lifespan.lower, buf.lifespan.upper},
        .size = buf.size,
        .alignment = buf.alignment,
        .gaps = {},
        .offset = std::nullopt,
        .hint = std::nullopt});
  }

  // Try to solve
  auto result = solver.Solve(problem);
  if (!result.ok() ||
      result->offsets.size() != tier.buffers.size()) {
    // Failed - remove the buffer we just added
    tier.buffers.pop_back();
    llvm::errs() << "[tryAllocate] " << tier.name << ": FAILED for id="
                 << buffer.id;
    if (!result.ok()) {
      llvm::errs() << " (solver error: " << result.status().message() << ")";
    } else {
      llvm::errs() << " (offset count mismatch: got "
                   << result->offsets.size() << " expected "
                   << tier.buffers.size() + 1 << ")";
    }
    llvm::errs() << "\n";
    return std::nullopt;
  }

  llvm::errs() << "[tryAllocate] " << tier.name << ": SUCCESS for id="
               << buffer.id << " offsets=[";
  for (size_t i = 0; i < result->offsets.size(); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << tier.buffers[i].id << ":" << result->offsets[i];
  }
  llvm::errs() << "]\n";

  return *result;
}

/// Remove expired intervals from a tier.
void expireTierIntervals(const LiveInterval &cur, MemoryTier &tier,
                         llvm::StringMap<int64_t> &memMap) {
  llvm::SmallVector<size_t> expiredIndices;

  // Find expired intervals
  for (size_t i = 0; i < tier.active.size(); ++i) {
    if (tier.active[i].end <= cur.start) {
      expiredIndices.push_back(i);
    }
  }

  if (!expiredIndices.empty()) {
    llvm::errs() << "[expire] " << tier.name << ": expiring "
                 << expiredIndices.size() << " intervals at cur.start="
                 << cur.start << ": ";
    for (size_t idx : expiredIndices) {
      llvm::errs() << tier.active[idx].id << "(end=" << tier.active[idx].end << ") ";
    }
    llvm::errs() << "\n";
  }

  // Remove in reverse order to maintain valid indices
  for (auto it = expiredIndices.rbegin(); it != expiredIndices.rend(); ++it) {
    tier.handled.push_back(std::move(tier.active[*it]));
    tier.active.erase(tier.active.begin() + *it);
    // Note: we don't remove from memMap here as Python does pop with default
  }
}

/// Handle successful allocation - update offsets and memMap.
void allocSuccess(LiveInterval &cur, MemoryTier &tier,
                  const minimalloc::Solution &solution,
                  llvm::StringMap<int64_t> &memMap) {
  llvm::errs() << "[allocSuccess] " << tier.name << ": allocated id=" << cur.id
               << " [" << cur.start << ", " << cur.end << ") size=" << cur.size
               << "\n";

  tier.active.push_back(cur);

  // Update offsets for all buffers and their corresponding LiveIntervals
  for (size_t i = 0; i < tier.buffers.size(); ++i) {
    int64_t offset = solution.offsets[i];
    tier.buffers[i].offset = offset;

    // Find and update matching active interval
    for (auto &iv : tier.active) {
      if (iv.id == tier.buffers[i].id) {
        iv.offset = offset;
        break;
      }
    }

    memMap[tier.buffers[i].id] = offset;
  }

  llvm::errs() << "[allocSuccess] " << tier.name << ": memMap after: {";
  for (const auto &kv : memMap) {
    llvm::errs() << kv.first() << ":" << kv.second << " ";
  }
  llvm::errs() << "}\n";
  llvm::errs() << "[allocSuccess] " << tier.name << ": active intervals ("
               << tier.active.size() << "): ";
  for (const auto &iv : tier.active) {
    llvm::errs() << "{id=" << iv.id << " [" << iv.start << "," << iv.end
                 << ") off=" << iv.offset << "} ";
  }
  llvm::errs() << "\n";
}

/// Select which interval to spill from a tier.
std::optional<std::string> selectSpillVictim(const LiveInterval &cur,
                                              MemoryTier &tier,
                                              minimalloc::Solver &solver) {
  llvm::errs() << "[selectSpillVictim] " << tier.name
               << ": finding victim for cur.id=" << cur.id
               << " cur.start=" << cur.start << "\n";

  // Create problem to find IIS
  minimalloc::Problem problem;
  problem.capacity = tier.capacity;
  for (const auto &buf : tier.buffers) {
    problem.buffers.push_back(minimalloc::Buffer{
        .id = buf.id,
        .lifespan = {buf.lifespan.lower, buf.lifespan.upper},
        .size = buf.size,
        .alignment = buf.alignment,
        .gaps = {},
        .offset = std::nullopt,
        .hint = std::nullopt});
  }

  auto subset = solver.ComputeIrreducibleInfeasibleSubset(problem);
  if (!subset.ok() || subset->empty()) {
    llvm::errs() << "[selectSpillVictim] " << tier.name
                 << ": IIS computation failed or empty";
    if (!subset.ok()) {
      llvm::errs() << " (error: " << subset.status().message() << ")";
    }
    llvm::errs() << "\n";
    return std::nullopt;
  }

  llvm::errs() << "[selectSpillVictim] " << tier.name << ": IIS indices=[";
  for (size_t i = 0; i < subset->size(); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << (*subset)[i];
  }
  llvm::errs() << "]\n";

  auto conflictingIntervals =
      getIntervalsFromSubset(*subset, tier.buffers, tier.active);
  if (conflictingIntervals.empty()) {
    llvm::errs() << "[selectSpillVictim] " << tier.name
                 << ": no conflicting intervals found in active list\n";
    llvm::errs() << "[selectSpillVictim] " << tier.name
                 << ": active list (" << tier.active.size() << "): ";
    for (const auto &iv : tier.active) {
      llvm::errs() << iv.id << " ";
    }
    llvm::errs() << "\n";
    llvm::errs() << "[selectSpillVictim] " << tier.name
                 << ": buffer list (" << tier.buffers.size() << "): ";
    for (const auto &buf : tier.buffers) {
      llvm::errs() << buf.id << " ";
    }
    llvm::errs() << "\n";
    return std::nullopt;
  }

  llvm::errs() << "[selectSpillVictim] " << tier.name
               << ": conflicting intervals: ";
  for (auto *iv : conflictingIntervals) {
    llvm::errs() << "{id=" << iv->id << " uses=[";
    for (size_t i = 0; i < iv->uses.size(); ++i) {
      if (i > 0) llvm::errs() << ",";
      llvm::errs() << iv->uses[i];
    }
    llvm::errs() << "] end=" << iv->end << "} ";
  }
  llvm::errs() << "\n";

  // Weight by distance to next use (prefer spilling intervals with distant next use)
  llvm::StringMap<int64_t> weight;
  for (auto *iv : conflictingIntervals) {
    llvm::SmallVector<int64_t> usesCopy = iv->uses;
    usesCopy.push_back(iv->end);

    for (int64_t use : usesCopy) {
      int64_t diff = use - cur.start;
      if (diff > 0) {
        weight[iv->id] = diff;
        break;
      }
    }
  }

  if (weight.empty()) {
    llvm::errs() << "[selectSpillVictim] " << tier.name
                 << ": no positive weights found, cannot select victim\n";
    return std::nullopt;
  }

  // Find the id with maximum weight
  std::string maxId;
  int64_t maxWeight = -1;
  for (const auto &kv : weight) {
    if (kv.second > maxWeight) {
      maxWeight = kv.second;
      maxId = kv.first().str();
    }
  }

  llvm::errs() << "[selectSpillVictim] " << tier.name << ": weights={";
  for (const auto &kv : weight) {
    llvm::errs() << kv.first() << ":" << kv.second << " ";
  }
  llvm::errs() << "} -> selected victim=" << maxId << " (weight=" << maxWeight
               << ")\n";

  return maxId;
}

/// Compute the spill for a given interval.
SpillResult computeSpill(const LiveInterval &cur, MemoryTier &tier,
                         const std::string &spillId) {
  llvm::errs() << "[computeSpill] " << tier.name << ": spilling id=" << spillId
               << " at cur.start=" << cur.start << "\n";

  // Remove the interval from active
  auto spilledIntervalOpt = tier.removeActive(spillId);
  if (!spilledIntervalOpt) {
    llvm::errs() << "[computeSpill] " << tier.name
                 << ": FAILED - could not find id=" << spillId
                 << " in active list\n";
    return SpillResult();
  }
  LiveInterval spilledInterval = std::move(*spilledIntervalOpt);

  // Remove the buffer
  auto removedBufferOpt = tier.removeBuffer(spillId);
  if (!removedBufferOpt) {
    llvm::errs() << "[computeSpill] " << tier.name
                 << ": FAILED - could not find buffer id=" << spillId << "\n";
    return SpillResult();
  }

  int64_t originalStart = spilledInterval.start;
  int64_t originalEnd = spilledInterval.end;
  int64_t spillStart = cur.start;

  llvm::errs() << "[computeSpill] " << tier.name
               << ": spilled interval [" << originalStart << ", "
               << originalEnd << ") size=" << spilledInterval.size
               << " uses=[";
  for (size_t i = 0; i < spilledInterval.uses.size(); ++i) {
    if (i > 0) llvm::errs() << ",";
    llvm::errs() << spilledInterval.uses[i];
  }
  llvm::errs() << "]\n";

  // Find next use after spillStart
  std::optional<int64_t> nextUse;
  for (int64_t u : spilledInterval.uses) {
    if (u > spillStart) {
      nextUse = u;
      break;
    }
  }
  if (!nextUse) {
    nextUse = originalEnd - 1;
  }

  llvm::errs() << "[computeSpill] " << tier.name
               << ": spillStart=" << spillStart << " nextUse=" << *nextUse
               << "\n";

  // Prefix: portion BEFORE spill point (stays in this tier)
  std::optional<Buffer> prefixBuffer;
  std::optional<LiveInterval> prefixInterval;
  if (originalStart < spillStart) {
    prefixBuffer = Buffer(spillId, Interval(originalStart, spillStart),
                          spilledInterval.size, 1);

    llvm::SmallVector<int64_t> prefixUses;
    for (int64_t u : spilledInterval.uses) {
      if (u < spillStart)
        prefixUses.push_back(u);
    }
    prefixInterval = LiveInterval(spillId, spilledInterval.size, originalStart,
                                  spillStart, std::move(prefixUses));
    llvm::errs() << "[computeSpill] " << tier.name << ": prefix=["
                 << originalStart << ", " << spillStart << ")\n";
  } else {
    llvm::errs() << "[computeSpill] " << tier.name << ": no prefix\n";
  }

  // Spill: goes to next tier
  std::optional<Buffer> spillBuffer;
  if (spillStart < *nextUse) {
    spillBuffer =
        Buffer(spillId, Interval(spillStart, *nextUse), spilledInterval.size, 1);
    llvm::errs() << "[computeSpill] " << tier.name << ": spill=["
                 << spillStart << ", " << *nextUse << ")\n";
  } else {
    llvm::errs() << "[computeSpill] " << tier.name << ": no spill buffer\n";
  }

  // Reload: comes back to this tier
  std::optional<LiveInterval> reloadInterval;
  if (*nextUse < originalEnd) {
    llvm::SmallVector<int64_t> reloadUses;
    for (int64_t u : spilledInterval.uses) {
      if (u >= *nextUse)
        reloadUses.push_back(u);
    }
    reloadInterval = LiveInterval(spillId, spilledInterval.size, *nextUse,
                                  originalEnd, std::move(reloadUses));
    llvm::errs() << "[computeSpill] " << tier.name << ": reload=["
                 << *nextUse << ", " << originalEnd << ")\n";
  } else {
    llvm::errs() << "[computeSpill] " << tier.name << ": no reload\n";
  }

  return SpillResult(std::move(reloadInterval), std::move(spillBuffer),
                     std::move(prefixBuffer), std::move(prefixInterval));
}

// Forward declarations
void processTier(size_t tierIdx, llvm::SmallVector<MemoryTier, 4> &tiers,
                 minimalloc::Solver &solver, llvm::StringMap<int64_t> &memMap);

bool allocateAtTier(LiveInterval &cur, size_t tierIdx,
                    llvm::SmallVector<MemoryTier, 4> &tiers,
                    minimalloc::Solver &solver,
                    llvm::StringMap<int64_t> &memMap);

/// Handle spilling from a tier to the next tier.
bool handleSpill(const LiveInterval &cur, size_t tierIdx,
                 llvm::SmallVector<MemoryTier, 4> &tiers,
                 minimalloc::Solver &solver,
                 llvm::StringMap<int64_t> &memMap) {
  MemoryTier &tier = tiers[tierIdx];
  size_t nextTierIdx = tierIdx + 1;

  llvm::errs() << "[handleSpill] " << tier.name << ": spilling for cur.id="
               << cur.id << " [" << cur.start << ", " << cur.end << ")\n";

  if (nextTierIdx >= tiers.size()) {
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": FAILED - no next tier available\n";
    return false;
  }

  MemoryTier &nextTier = tiers[nextTierIdx];

  // Select which interval to spill
  auto spillIdOpt = selectSpillVictim(cur, tier, solver);
  if (!spillIdOpt) {
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": FAILED - no spill victim found\n";
    return false;
  }

  llvm::errs() << "[handleSpill] " << tier.name << ": spilling victim="
               << *spillIdOpt << " to " << nextTier.name << "\n";

  // Compute the spill
  SpillResult result = computeSpill(cur, tier, *spillIdOpt);

  // Handle prefix portion (stays in current tier)
  if (result.prefixBuffer && result.prefixInterval) {
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": allocating prefix for " << result.prefixBuffer->id
                 << " [" << result.prefixBuffer->lifespan.lower << ", "
                 << result.prefixBuffer->lifespan.upper << ")\n";
    auto prefixSolution = tryAllocate(tier, *result.prefixBuffer, solver);
    if (prefixSolution) {
      tier.active.push_back(*result.prefixInterval);
      llvm::errs() << "[handleSpill] " << tier.name
                   << ": prefix allocated successfully\n";
    } else {
      llvm::errs() << "[handleSpill] " << tier.name
                   << ": prefix allocation FAILED\n";
    }
  }

  // Handle spill portion (goes to next tier)
  if (result.spillBuffer) {
    LiveInterval spillInterval(result.spillBuffer->id, result.spillBuffer->size,
                               result.spillBuffer->lifespan.lower,
                               result.spillBuffer->lifespan.upper);
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": adding spill to " << nextTier.name
                 << " id=" << spillInterval.id << " ["
                 << spillInterval.start << ", " << spillInterval.end << ")\n";
    nextTier.addUnhandled(std::move(spillInterval));

    // Process the next tier to allocate the spill
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": recursing into " << nextTier.name << "\n";
    processTier(nextTierIdx, tiers, solver, memMap);
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": returned from " << nextTier.name << "\n";
  }

  // Handle reload portion (comes back to this tier)
  if (result.reloadInterval && result.reloadInterval->start < result.reloadInterval->end) {
    Buffer *dstBuffer = nextTier.getBuffer(*spillIdOpt);
    if (dstBuffer) {
      result.reloadInterval->reloadFromTier = nextTierIdx;
      result.reloadInterval->reloadFromOffset = dstBuffer->offset;
      llvm::errs() << "[handleSpill] " << tier.name
                   << ": reload from " << nextTier.name
                   << " offset=" << dstBuffer->offset << "\n";
    } else {
      llvm::errs() << "[handleSpill] " << tier.name
                   << ": WARNING - could not find spill buffer in "
                   << nextTier.name << " for id=" << *spillIdOpt << "\n";
    }
    llvm::errs() << "[handleSpill] " << tier.name
                 << ": adding reload to unhandled id="
                 << result.reloadInterval->id << " ["
                 << result.reloadInterval->start << ", "
                 << result.reloadInterval->end << ")\n";
    tier.addUnhandled(std::move(*result.reloadInterval));
  }

  llvm::errs() << "[handleSpill] " << tier.name << ": spill complete\n";
  return true;
}

/// Try to allocate an interval at a specific tier.
bool allocateAtTier(LiveInterval &cur, size_t tierIdx,
                    llvm::SmallVector<MemoryTier, 4> &tiers,
                    minimalloc::Solver &solver,
                    llvm::StringMap<int64_t> &memMap) {
  if (tierIdx >= tiers.size()) {
    llvm::errs() << "[allocateAtTier] tierIdx=" << tierIdx
                 << " out of range (num tiers=" << tiers.size() << ")\n";
    return false;
  }

  MemoryTier &tier = tiers[tierIdx];

  llvm::errs() << "[allocateAtTier] " << tier.name << ": attempting id="
               << cur.id << " [" << cur.start << ", " << cur.end
               << ") size=" << cur.size << "\n";

  Buffer buffer(cur.id, Interval(cur.start, cur.end), cur.size, 1);

  auto solution = tryAllocate(tier, buffer, solver);
  if (solution) {
    allocSuccess(cur, tier, *solution, memMap);
    return true;
  }

  llvm::errs() << "[allocateAtTier] " << tier.name
               << ": first attempt failed, trying spill\n";

  // Allocation failed - spill and retry
  if (!handleSpill(cur, tierIdx, tiers, solver, memMap)) {
    llvm::errs() << "[allocateAtTier] " << tier.name
                 << ": spill FAILED for id=" << cur.id << "\n";
    return false;
  }

  llvm::errs() << "[allocateAtTier] " << tier.name
               << ": retrying allocation after spill for id=" << cur.id << "\n";

  solution = tryAllocate(tier, buffer, solver);
  if (solution) {
    allocSuccess(cur, tier, *solution, memMap);
    return true;
  }

  llvm::errs() << "[allocateAtTier] " << tier.name
               << ": FAILED even after spill for id=" << cur.id << "\n";
  return false;
}

/// Process all intervals in a tier's unhandled list.
void processTier(size_t tierIdx, llvm::SmallVector<MemoryTier, 4> &tiers,
                 minimalloc::Solver &solver,
                 llvm::StringMap<int64_t> &memMap) {
  MemoryTier &tier = tiers[tierIdx];

  llvm::errs() << "\n=== [processTier] " << tier.name
               << " (level=" << tier.level << " capacity=" << tier.capacity
               << ") unhandled=" << tier.unhandled.size() << " ===\n";
  for (const auto &iv : tier.unhandled) {
    llvm::errs() << "  unhandled: id=" << iv.id << " [" << iv.start << ", "
                 << iv.end << ") size=" << iv.size << " uses=[";
    for (size_t i = 0; i < iv.uses.size(); ++i) {
      if (i > 0) llvm::errs() << ",";
      llvm::errs() << iv.uses[i];
    }
    llvm::errs() << "]\n";
  }

  while (!tier.unhandled.empty()) {
    LiveInterval cur = std::move(tier.unhandled.front());
    tier.unhandled.erase(tier.unhandled.begin());

    llvm::errs() << "\n--- [processTier] " << tier.name << ": processing id="
                 << cur.id << " [" << cur.start << ", " << cur.end
                 << ") size=" << cur.size << " ---\n";

    // Expire intervals that have ended
    expireTierIntervals(cur, tier, memMap);

    // Skip invalid intervals
    if (cur.start >= cur.end) {
      llvm::errs() << "[processTier] " << tier.name << ": SKIPPING id="
                   << cur.id << " (start=" << cur.start
                   << " >= end=" << cur.end << ")\n";
      continue;
    }

    // Try to allocate at this tier
    allocateAtTier(cur, tierIdx, tiers, solver, memMap);
  }

  llvm::errs() << "=== [processTier] " << tier.name << ": done ===\n\n";
}

/// Run linear scan memory allocation with recursive spilling across N tiers.
llvm::StringMap<int64_t>
allocate(const AllocationProblem &problem,
         llvm::ArrayRef<int64_t> tierCapacities = {12288, 16384, 131072}) {

  llvm::errs() << "\n╔══════════════════════════════════════════╗\n"
               << "║  ALLOCATION START                        ║\n"
               << "╚══════════════════════════════════════════╝\n";
  llvm::errs() << "Intervals (" << problem.intervals.size() << "):\n";
  for (const auto &iv : problem.intervals) {
    llvm::errs() << "  id=" << iv.id << " size=" << iv.size << " ["
                 << iv.start << ", " << iv.end << ") uses=[";
    for (size_t i = 0; i < iv.uses.size(); ++i) {
      if (i > 0) llvm::errs() << ",";
      llvm::errs() << iv.uses[i];
    }
    llvm::errs() << "]\n";
  }
  llvm::errs() << "Tier capacities: [";
  for (size_t i = 0; i < tierCapacities.size(); ++i) {
    if (i > 0) llvm::errs() << ", ";
    llvm::errs() << tierCapacities[i];
  }
  llvm::errs() << "]\n\n";

  // Create memory tiers (tier 0 = fastest/smallest)
  llvm::SmallVector<MemoryTier, 4> tiers;
  for (auto [i, cap] : llvm::enumerate(tierCapacities)) {
    std::string name = "Tier " + std::to_string(tierCapacities.size() - i);
    tiers.emplace_back(std::move(name), cap, static_cast<int64_t>(i));
    llvm::errs() << "Created " << tiers.back().name << " (level=" << i
                 << " capacity=" << cap << ")\n";
  }

  minimalloc::Solver solver;
  llvm::StringMap<int64_t> memMap;

  // Load intervals into top tier's unhandled list
  tiers[0].unhandled = problem.sortedIntervals();

  // Process the top tier (recursively processes lower tiers as needed)
  processTier(0, tiers, solver, memMap);

  llvm::errs() << "\n╔══════════════════════════════════════════╗\n"
               << "║  ALLOCATION COMPLETE                     ║\n"
               << "╚══════════════════════════════════════════╝\n";
  llvm::errs() << "Final memMap (" << memMap.size() << " entries):\n";
  for (const auto &kv : memMap) {
    llvm::errs() << "  id=" << kv.first() << " -> offset=" << kv.second << "\n";
  }
  llvm::errs() << "\n";

  return memMap;
}



class LocalStagingPass
    : public impl::LocalStagingBase<LocalStagingPass> {
public:
  using LocalStagingBase::LocalStagingBase;

  void runOnOperation() override {

    // Lifetime container
    llvm::SmallVector<LiveInterval> lifetimeInfos;

    ModuleOp module = getOperation();

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
        llvm::errs() << "  SSA value: " << memref << "\n";
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


    AllocationProblem problem;

    for (auto &interval : lifetimeInfos) {
      problem.add(interval);
    }

    auto map = allocate(problem);

    for (auto &index : map) {
      llvm::StringRef key = index.getKey();
      int value = index.getValue();
      llvm::errs() << "MAP ID" << key << "MAP INT" << value  << "\n"; 
    }


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
