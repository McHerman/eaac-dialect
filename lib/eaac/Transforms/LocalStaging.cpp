//===- LocalStaging.cpp - Insert local SRAM staging for compute ops -===//
//
// Pass to insert memref.copy operations to stage data into local SRAM
// before compute operations and copy results back afterward.
//
//===----------------------------------------------------------------------===//

#include "eaac/MemRefLivenessAnalysis.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "minimalloc.h"
#include "solver.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#define DEBUG_TYPE "local-staging"

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

using eaac::LiveInterval;
using eaac::MemRefUse;

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
  Value spilledMemref; // The victim's original memref Value

  SpillResult() = default;
  SpillResult(std::optional<LiveInterval> reloadInterval,
              std::optional<Buffer> spillBuffer,
              Value spilledMemref = Value())
      : reloadInterval(std::move(reloadInterval)),
        spillBuffer(std::move(spillBuffer)),
        spilledMemref(spilledMemref) {}
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
  LLVM_DEBUG({
    llvm::dbgs() << "[tryAllocate] " << tier.name << ": trying id=" << buffer.id
                 << " size=" << buffer.size << " [" << buffer.lifespan.lower
                 << ", " << buffer.lifespan.upper << ") capacity="
                 << tier.capacity << " existing=(" << tier.buffers.size()
                 << "): ";
    for (const auto &buf : tier.buffers)
      llvm::dbgs() << buf.id << " ";
    llvm::dbgs() << "\n";
  });

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
        .offset = buf.offset >= 0 ? std::optional<int64_t>(buf.offset)
                                   : std::nullopt,
        .hint = std::nullopt});
  }

  // Try to solve
  auto result = solver.Solve(problem);
  if (!result.ok() ||
      result->offsets.size() != tier.buffers.size()) {
    // Failed - remove the buffer we just added
    tier.buffers.pop_back();
    LLVM_DEBUG({
      llvm::dbgs() << "[tryAllocate] " << tier.name << ": FAILED for id="
                   << buffer.id;
      if (!result.ok())
        llvm::dbgs() << " (" << result.status().message() << ")";
      llvm::dbgs() << "\n";
    });
    return std::nullopt;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[tryAllocate] " << tier.name << ": SUCCESS id="
                 << buffer.id << " offsets=[";
    for (size_t i = 0; i < result->offsets.size(); ++i) {
      if (i > 0) llvm::dbgs() << ", ";
      llvm::dbgs() << tier.buffers[i].id << ":" << result->offsets[i];
    }
    llvm::dbgs() << "]\n";
  });

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

  LLVM_DEBUG({
    if (!expiredIndices.empty()) {
      llvm::dbgs() << "[expire] " << tier.name << ": expiring "
                   << expiredIndices.size() << " at cur.start=" << cur.start
                   << ":";
      for (size_t idx : expiredIndices)
        llvm::dbgs() << " " << tier.active[idx].id;
      llvm::dbgs() << "\n";
    }
  });

  // Remove in reverse order to maintain valid indices
  for (auto it = expiredIndices.rbegin(); it != expiredIndices.rend(); ++it) {
    tier.removeBuffer(tier.active[*it].id);
    tier.handled.push_back(std::move(tier.active[*it]));
    tier.active.erase(tier.active.begin() + *it);
  }
}

/// Handle successful allocation - update offsets and memMap.
void allocSuccess(LiveInterval &cur, MemoryTier &tier,
                  const minimalloc::Solution &solution,
                  llvm::StringMap<int64_t> &memMap) {
  LLVM_DEBUG(llvm::dbgs() << "[allocSuccess] " << tier.name << ": id=" << cur.id
                          << " [" << cur.start << ", " << cur.end
                          << ") size=" << cur.size << "\n");

  tier.active.push_back(cur);

  // Update offsets for all buffers and their corresponding LiveIntervals
  //
  //
  // TODO, this might actually be a little iffy, 
  // since the buffers probably shouldnt have their offset recalculated after assgingments
  for (size_t i = 0; i < tier.buffers.size(); ++i) {
    int64_t offset = solution.offsets[i];
    tier.buffers[i].offset = offset;

    // Find and update matching active interval
    for (auto &iv : tier.active) {
      if (iv.id == tier.buffers[i].id) {
        iv.offset = offset;
        // Annotate only the newly allocated interval's memref
        if (iv.id == cur.id) {
          // TODO, move this logic to a function instead
          if (Operation *defOp = iv.memref.getDefiningOp()) {
            OpBuilder builder(defOp);
            defOp->setAttr("eaac.offset",
                           builder.getI64IntegerAttr(offset));
            defOp->setAttr("eaac.tier",
                           builder.getI64IntegerAttr(tier.level));
          }
        }
        break;
      }
    }

    memMap[tier.buffers[i].id] = offset;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "[allocSuccess] " << tier.name << ": active=";
    for (const auto &iv : tier.active)
      llvm::dbgs() << "{" << iv.id << " off=" << iv.offset << "} ";
    llvm::dbgs() << "\n";
  });
}

/// Select which interval to spill from a tier.
std::optional<std::string> selectSpillVictim(const LiveInterval &cur,
                                              MemoryTier &tier,
                                              minimalloc::Solver &solver) {
  LLVM_DEBUG(llvm::dbgs() << "[selectSpillVictim] " << tier.name
                          << ": for cur.id=" << cur.id << "\n");

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
        .offset = buf.offset >= 0 ? std::optional<int64_t>(buf.offset)
                                   : std::nullopt,
        .hint = std::nullopt});
  }

  auto subset = solver.ComputeIrreducibleInfeasibleSubset(problem);
  if (!subset.ok() || subset->empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[selectSpillVictim] " << tier.name
                            << ": IIS failed or empty\n");
    return std::nullopt;
  }

  auto conflictingIntervals =
      getIntervalsFromSubset(*subset, tier.buffers, tier.active);
  if (conflictingIntervals.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "[selectSpillVictim] " << tier.name
                   << ": no conflicts in active (active=" << tier.active.size()
                   << " buffers=" << tier.buffers.size() << ")\n";
    });
    return std::nullopt;
  }

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
    LLVM_DEBUG(llvm::dbgs() << "[selectSpillVictim] " << tier.name
                            << ": no positive weights\n");
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

  LLVM_DEBUG(llvm::dbgs() << "[selectSpillVictim] " << tier.name
                          << ": victim=" << maxId << " weight=" << maxWeight
                          << "\n");

  return maxId;
}

/// Compute the spill for a given interval.
/// Truncates the victim's active interval and buffer in-place to end at
/// spillStart, and returns spill/reload intervals for the remaining portion.
SpillResult computeSpill(const LiveInterval &cur, MemoryTier &tier,
                         const std::string &spillId) {
  LLVM_DEBUG(llvm::dbgs() << "[computeSpill] " << tier.name << ": id="
                          << spillId << " at cur.start=" << cur.start << "\n");

  // Find the active interval and buffer by reference (keep them in place)
  LiveInterval *activeInterval = nullptr;
  for (auto &iv : tier.active) {
    if (iv.id == spillId) {
      activeInterval = &iv;
      break;
    }
  }
  if (!activeInterval) {
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] FAILED: " << spillId
                            << " not in active\n");
    return SpillResult();
  }

  Buffer *buffer = tier.getBuffer(spillId);
  if (!buffer) {
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] FAILED: buffer " << spillId
                            << " not found\n");
    return SpillResult();
  }

  int64_t originalEnd = activeInterval->end;
  int64_t spillStart = cur.start;
  Value originalMemref = activeInterval->memref;

  // Save original uses before truncation (needed for spill/reload construction)
  llvm::SmallVector<int64_t> originalUses = activeInterval->uses;

  // Find next use after spillStart
  std::optional<int64_t> nextUse;
  for (int64_t u : originalUses) {
    if (u > spillStart) {
      nextUse = u;
      break;
    }
  }
  if (!nextUse) {
    nextUse = originalEnd - 1;
  }

  LLVM_DEBUG(llvm::dbgs() << "[computeSpill] " << tier.name << ": ["
                          << activeInterval->start << ", " << originalEnd
                          << ") spillStart=" << spillStart
                          << " nextUse=" << *nextUse << "\n");

  // Truncate the victim in-place to keep only the prefix portion
  activeInterval->end = spillStart;
  llvm::erase_if(activeInterval->uses,
                 [spillStart](int64_t u) { return u >= spillStart; });
  buffer->lifespan.upper = spillStart;

  LLVM_DEBUG(llvm::dbgs() << "[computeSpill] truncated to ["
                          << activeInterval->start << ", " << spillStart
                          << ")\n");

  // Spill: goes to next tier
  std::optional<Buffer> spillBuffer;
  if (spillStart < *nextUse) {
    spillBuffer =
        Buffer(spillId, Interval(spillStart, *nextUse), activeInterval->size, 1);
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] spill=[" << spillStart << ", "
                            << *nextUse << ")\n");
  }

  // Reload: comes back to this tier
  std::optional<LiveInterval> reloadInterval;
  if (*nextUse < originalEnd) {
    llvm::SmallVector<int64_t> reloadUses;
    for (int64_t u : originalUses) {
      if (u >= *nextUse)
        reloadUses.push_back(u);
    }
    reloadInterval = LiveInterval(spillId, activeInterval->size, *nextUse,
                                  originalEnd, std::move(reloadUses));
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] reload=[" << *nextUse << ", "
                            << originalEnd << ")\n");
  }

  return SpillResult(std::move(reloadInterval), std::move(spillBuffer),
                     originalMemref);
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

  LLVM_DEBUG(llvm::dbgs() << "[handleSpill] " << tier.name << ": cur.id="
                          << cur.id << " [" << cur.start << ", " << cur.end
                          << ")\n");

  if (nextTierIdx >= tiers.size()) {
    LLVM_DEBUG(llvm::dbgs() << "[handleSpill] no next tier\n");
    return false;
  }

  MemoryTier &nextTier = tiers[nextTierIdx];

  // Select which interval to spill
  auto spillIdOpt = selectSpillVictim(cur, tier, solver);
  if (!spillIdOpt) {
    LLVM_DEBUG(llvm::dbgs() << "[handleSpill] no victim found\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "[handleSpill] victim=" << *spillIdOpt << " -> "
                          << nextTier.name << "\n");

  // Compute the spill (truncates victim in-place, returns spill/reload)
  SpillResult result = computeSpill(cur, tier, *spillIdOpt);

  // Create IR operations for spill and reload buffers
  bool hasSpill = result.spillBuffer.has_value();
  bool hasReload = result.reloadInterval.has_value() &&
                   result.reloadInterval->start < result.reloadInterval->end;

  Value spillMemref;
  Value reloadMemref;

  //TODO, conditionals should prob not be here

  if (hasSpill || hasReload) {
    Value originalMemref = result.spilledMemref;
    auto type = llvm::cast<MemRefType>(originalMemref.getType());
    Location loc = originalMemref.getLoc();

    // TODO, make this a function call 
    // BUG, this should be before the CUR opt, not after the original.
    //OpBuilder builder(originalMemref.getDefiningOp());
    //builder.setInsertionPointAfter(originalMemref.getDefiningOp());
    
    OpBuilder builder(cur.memref.getDefiningOp());
    builder.setInsertionPoint(cur.memref.getDefiningOp());

    // Spill: alloc + copy original -> spill buffer
    if (hasSpill) {
      spillMemref = memref::AllocOp::create(builder, loc, type);
      //memref::CopyOp::create(builder, loc, originalMemref, spillMemref);
    }

    OpBuilder builder2(cur.memref.getDefiningOp());
    builder2.setInsertionPointAfter(cur.memref.getDefiningOp());

    // Reload: alloc + copy spill -> reload buffer
    // BUG This is not correct, that should be placed at next use 
    if (hasReload) {
      // TODO, completely unessecary check.
      if (spillMemref) {
        reloadMemref = memref::AllocOp::create(builder, loc, type);
        memref::CopyOp::create(builder, loc, spillMemref, reloadMemref);
        // Spill buffer served its purpose, free it
        memref::DeallocOp::create(builder, loc, spillMemref);
      } else {
        // No spill window — reuse original memref
        reloadMemref = originalMemref;
      }
    }
  }

  // Add spill interval to next tier
  // TODO, also completely unessecary check
  if (hasSpill) {
    LiveInterval spillInterval(result.spillBuffer->id, result.spillBuffer->size,
                               result.spillBuffer->lifespan.lower,
                               result.spillBuffer->lifespan.upper,
                               /*uses=*/{}, spillMemref);
    nextTier.addUnhandled(std::move(spillInterval));
    processTier(nextTierIdx, tiers, solver, memMap);
  }

  // Add reload interval back to current tier
  if (hasReload) {
    Buffer *dstBuffer = nextTier.getBuffer(*spillIdOpt);
    if (dstBuffer) {
      result.reloadInterval->reloadFromTier = nextTierIdx;
      result.reloadInterval->reloadFromOffset = dstBuffer->offset;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "[handleSpill] WARNING: spill buffer for "
                              << *spillIdOpt << " not found in "
                              << nextTier.name << "\n");
    }
    result.reloadInterval->memref = reloadMemref;
    tier.addUnhandled(std::move(*result.reloadInterval));
  }

  return true;
}

/// Try to allocate an interval at a specific tier.
bool allocateAtTier(LiveInterval &cur, size_t tierIdx,
                    llvm::SmallVector<MemoryTier, 4> &tiers,
                    minimalloc::Solver &solver,
                    llvm::StringMap<int64_t> &memMap) {
  if (tierIdx >= tiers.size())
    return false;

  MemoryTier &tier = tiers[tierIdx];
  LLVM_DEBUG(llvm::dbgs() << "[allocateAtTier] " << tier.name << ": id="
                          << cur.id << " [" << cur.start << ", " << cur.end
                          << ") size=" << cur.size << "\n");

  Buffer buffer(cur.id, Interval(cur.start, cur.end), cur.size, 1);

  auto solution = tryAllocate(tier, buffer, solver);
  if (solution) {
    allocSuccess(cur, tier, *solution, memMap);
    return true;
  }

  // Try a single spill and retry
  LLVM_DEBUG(llvm::dbgs() << "[allocateAtTier] " << tier.name
                          << ": trying spill\n");
  if (!handleSpill(cur, tierIdx, tiers, solver, memMap)) {
    LLVM_DEBUG(llvm::dbgs() << "[allocateAtTier] " << tier.name
                            << ": FAILED, no spill victim for id=" << cur.id
                            << "\n");
    return false;
  }

  LLVM_DEBUG(llvm::dbgs() << "[allocateAtTier] " << tier.name
                          << ": retrying after spill\n");
  solution = tryAllocate(tier, buffer, solver);
  if (solution) {
    allocSuccess(cur, tier, *solution, memMap);
    return true;
  }

  LLVM_DEBUG(llvm::dbgs() << "[allocateAtTier] " << tier.name
                          << ": FAILED after spill for id=" << cur.id
                          << "\n");
  return false;
}

/// Process all intervals in a tier's unhandled list.
void processTier(size_t tierIdx, llvm::SmallVector<MemoryTier, 4> &tiers,
                 minimalloc::Solver &solver,
                 llvm::StringMap<int64_t> &memMap) {
  MemoryTier &tier = tiers[tierIdx];

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== processTier " << tier.name << " (capacity="
                 << tier.capacity << " unhandled=" << tier.unhandled.size()
                 << ") ===\n";
    for (const auto &iv : tier.unhandled)
      llvm::dbgs() << "  " << iv.id << " [" << iv.start << ", " << iv.end
                   << ") size=" << iv.size << "\n";
  });

  while (!tier.unhandled.empty()) {
    LiveInterval cur = std::move(tier.unhandled.front());
    tier.unhandled.erase(tier.unhandled.begin());

    LLVM_DEBUG(llvm::dbgs() << "\n--- " << tier.name << ": id=" << cur.id
                            << " [" << cur.start << ", " << cur.end << ") ---\n");

    // Expire intervals that have ended
    expireTierIntervals(cur, tier, memMap);

    // Skip invalid intervals
    if (cur.start >= cur.end) {
      LLVM_DEBUG(llvm::dbgs() << "  skipping (start >= end)\n");
      continue;
    }

    // Try to allocate at this tier
    allocateAtTier(cur, tierIdx, tiers, solver, memMap);
  }

  LLVM_DEBUG(llvm::dbgs() << "=== " << tier.name << " done ===\n\n");
}

/// Run linear scan memory allocation with recursive spilling across N tiers.
llvm::StringMap<int64_t>
allocate(const AllocationProblem &problem,
         llvm::ArrayRef<int64_t> tierCapacities = {49152, 147456, 16777216}) {

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== ALLOCATION START ===\nIntervals ("
                 << problem.intervals.size() << "):\n";
    for (const auto &iv : problem.intervals)
      llvm::dbgs() << "  id=" << iv.id << " size=" << iv.size << " ["
                   << iv.start << ", " << iv.end << ")\n";
    llvm::dbgs() << "Tiers: ";
    for (auto cap : tierCapacities)
      llvm::dbgs() << cap << " ";
    llvm::dbgs() << "\n";
  });

  // Create memory tiers (level N = fastest/smallest, level 1 = slowest/largest)
  llvm::SmallVector<MemoryTier, 4> tiers;
  for (auto [i, cap] : llvm::enumerate(tierCapacities)) {
    int64_t level = static_cast<int64_t>(tierCapacities.size() - i);
    std::string name = "Tier " + std::to_string(level);
    tiers.emplace_back(std::move(name), cap, level);
  }

  // Initialize minimalloc solver
  minimalloc::Solver solver;
  llvm::StringMap<int64_t> memMap;

  // Load intervals into top tier's unhandled list
  tiers[0].unhandled = problem.sortedIntervals();

  // Process the top tier (recursively processes lower tiers as needed)
  processTier(0, tiers, solver, memMap);

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== ALLOCATION COMPLETE ===\nFinal memMap ("
                 << memMap.size() << "):\n";
    for (const auto &kv : memMap)
      llvm::dbgs() << "  " << kv.first() << " -> " << kv.second << "\n";
  });

  return memMap;
}



class LocalStagingPass
    : public impl::LocalStagingBase<LocalStagingPass> {
public:
  using LocalStagingBase::LocalStagingBase;

  void runOnOperation() override {
    // Get lifetime analysis from the AnalysisManager
    auto &livenessAnalysis = getAnalysis<MemRefLivenessAnalysis>();
    const auto &lifetimeInfos = livenessAnalysis.getIntervals();

    AllocationProblem problem;
    for (const auto &interval : lifetimeInfos) {
      problem.add(interval);
    }

    auto map = allocate(problem);

  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLocalStagingPass() {
  return std::make_unique<LocalStagingPass>();
}

} // namespace eaac
} // namespace mlir
