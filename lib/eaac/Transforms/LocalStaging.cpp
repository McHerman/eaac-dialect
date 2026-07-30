//===- LocalStaging.cpp - Insert local SRAM staging for compute ops -===//
//
// Pass to insert memref.copy operations to stage data into local SRAM
// before compute operations and copy results back afterward.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/MemRefLivenessAnalysis.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "minimalloc.h"
#include "solver.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>

#define DEBUG_TYPE "local-staging"

// Debug helper callable from lldb: expr dumpIR(cur.memref)
LLVM_ATTRIBUTE_USED static void dumpIR(mlir::Value v) {
  if (auto *op = v.getDefiningOp())
    if (auto module = op->getParentOfType<mlir::ModuleOp>())
      module.dump();
}

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
  // Padding (in liveness time units) added to each buffer's lifespan upper
  // bound when handed to the solver. Forces an address gap between a buffer
  // that just died and the next buffer placed at the same offset, so anti-
  // aliasing through allocator reuse becomes impossible within R steps.
  int64_t reuseGuard = 0;
  // Required byte alignment for every buffer placed in this tier. Set from
  // the module's DLTI 'bus_size' entry; defaults to 1 (no alignment).
  int64_t alignment = 1;

  llvm::SmallVector<Buffer> buffers;
  llvm::SmallVector<LiveInterval> active;
  llvm::SmallVector<LiveInterval> handled;
  llvm::SmallVector<LiveInterval> unhandled;

  MemoryTier() : capacity(0), level(0) {}
  MemoryTier(std::string name, int64_t capacity, int64_t level,
             int64_t reuseGuard = 0, int64_t alignment = 1)
      : name(std::move(name)), capacity(capacity), level(level),
        reuseGuard(reuseGuard), alignment(alignment) {}

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
  Buffer spillBuffer;
  Value spilledMemref;
  std::optional<LiveInterval> reloadInterval;

  SpillResult(Buffer spillBuffer, Value spilledMemref,
              std::optional<LiveInterval> reloadInterval = std::nullopt)
      : spillBuffer(std::move(spillBuffer)),
        spilledMemref(spilledMemref),
        reloadInterval(std::move(reloadInterval)) {}
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


static std::optional<int64_t> getAlignment(ModuleOp module) {
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
    if (!key || key.getValue() != "bus_size")
      continue;
    auto intAttr = dyn_cast<IntegerAttr>(entry.getValue());
    if (!intAttr) {
      module.emitWarning() << "'bus_size': expected i64, got "
                           << entry.getValue() << "; ignoring";
      return std::nullopt;
    }
    return intAttr.getInt();
  }
  return std::nullopt;
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

  // Create minimalloc problem. Lifespan upper bounds are padded by
  // tier.reuseGuard so the solver enforces an address-reuse gap (see the
  // MemoryTier::reuseGuard comment).
  minimalloc::Problem problem;
  problem.capacity = tier.capacity;
  for (const auto &buf : tier.buffers) {
    problem.buffers.push_back(minimalloc::Buffer{
        .id = buf.id,
        .lifespan = {buf.lifespan.lower,
                     buf.lifespan.upper + tier.reuseGuard},
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

  // Find expired intervals. Respect tier.reuseGuard so a buffer stays in the
  // active/conflict set for R extra steps after its true death — otherwise
  // it'd be dropped before the next allocation could see it as a conflict
  // and the guard would have no effect.
  for (size_t i = 0; i < tier.active.size(); ++i) {
    if (tier.active[i].end + tier.reuseGuard <= cur.start) {
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
          if (Operation *defOp = iv.memref ? iv.memref.getDefiningOp()
                                            : nullptr) {
            OpBuilder builder(defOp);
            defOp->setAttr("eaac.offset",
                           builder.getI64IntegerAttr(offset));
            defOp->setAttr("eaac.tier",
                           builder.getI64IntegerAttr(tier.level));
            if (auto alloc = mlir::dyn_cast<memref::AllocOp>(defOp)) {
              auto oldType = alloc.getType();
              auto memSpace = eaac::MemSpaceAttr::get(
                  defOp->getContext(), tier.level, offset);
              alloc.getResult().setType(MemRefType::get(
                  oldType.getShape(), oldType.getElementType(),
                  oldType.getLayout(), memSpace));
            }
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

  // Create problem to find IIS. Pad lifespans consistently with tryAllocate
  // so the IIS reflects the same conflict graph the solver would see.
  minimalloc::Problem problem;
  problem.capacity = tier.capacity;
  for (const auto &buf : tier.buffers) {
    problem.buffers.push_back(minimalloc::Buffer{
        .id = buf.id,
        .lifespan = {buf.lifespan.lower,
                     buf.lifespan.upper + tier.reuseGuard},
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
    llvm::SmallVector<MemRefUse> usesCopy = iv->uses;
    usesCopy.push_back({nullptr, iv->end});

    for (const auto &use : usesCopy) {
      int64_t diff = use.time - cur.start;
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
std::optional<SpillResult> computeSpill(const LiveInterval &cur,
                                        MemoryTier &tier,
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
    return std::nullopt;
  }

  Buffer *buffer = tier.getBuffer(spillId);
  if (!buffer) {
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] FAILED: buffer " << spillId
                            << " not found\n");
    return std::nullopt;
  }

  int64_t originalEnd = activeInterval->end;
  int64_t spillStart = cur.start;
  Value originalMemref = activeInterval->memref;

  // Save original uses before truncation (needed for spill/reload construction)
  llvm::SmallVector<MemRefUse> originalUses = activeInterval->uses;

  // Find next use after spillStart
  std::optional<int64_t> nextUse;
  for (const auto &u : originalUses) {
    if (u.time > spillStart) {
      nextUse = u.time;
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
                 [spillStart](const MemRefUse &u) { return u.time >= spillStart; });
  buffer->lifespan.upper = spillStart;

  LLVM_DEBUG(llvm::dbgs() << "[computeSpill] truncated to ["
                          << activeInterval->start << ", " << spillStart
                          << ")\n");

  // Spill buffer always covers [spillStart, nextUse)
  Buffer spillBuffer(spillId, Interval(spillStart, *nextUse),
                     activeInterval->size, tier.alignment);
  LLVM_DEBUG(llvm::dbgs() << "[computeSpill] spill=[" << spillStart << ", "
                          << *nextUse << ")\n");

  // Reload: comes back to this tier (only if there's lifetime remaining)
  std::optional<LiveInterval> reloadInterval;
  if (*nextUse < originalEnd) {
    llvm::SmallVector<MemRefUse> reloadUses;
    for (const auto &u : originalUses) {
      if (u.time >= *nextUse)
        reloadUses.push_back(u);
    }
    reloadInterval = LiveInterval(spillId, activeInterval->size, *nextUse,
                                  originalEnd, std::move(reloadUses));
    LLVM_DEBUG(llvm::dbgs() << "[computeSpill] reload=[" << *nextUse << ", "
                            << originalEnd << ")\n");
  }

  return SpillResult(std::move(spillBuffer), originalMemref,
                     std::move(reloadInterval));
}

// Forward declarations
void processTier(size_t tierIdx, llvm::SmallVector<MemoryTier, 4> &tiers,
                 minimalloc::Solver &solver, llvm::StringMap<int64_t> &memMap);

bool allocateAtTier(LiveInterval &cur, size_t tierIdx,
                    llvm::SmallVector<MemoryTier, 4> &tiers,
                    minimalloc::Solver &solver,
                    llvm::StringMap<int64_t> &memMap);

/// Result of emitting an alloc + copy pair: a freshly-allocated memref and
/// the copy op that fills it from `src`. Used by both the spill path and the
/// home-tier staging-chain emission.
struct StagePoint {
  Value memref;
  memref::CopyOp copy;
};

/// Emit `%memref = memref.alloc; memref.copy %src, %memref` immediately
/// before `insertBefore`.
static StagePoint emitStage(OpBuilder &builder, Location loc, MemRefType type,
                            Value src, Operation *insertBefore) {
  builder.setInsertionPoint(insertBefore);
  Value memref = memref::AllocOp::create(builder, loc, type);
  auto copy = memref::CopyOp::create(builder, loc, src, memref);
  return {memref, copy};
}

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
  auto resultOpt = computeSpill(cur, tier, *spillIdOpt);
  if (!resultOpt) {
    LLVM_DEBUG(llvm::dbgs() << "[handleSpill] computeSpill failed\n");
    return false;
  }
  SpillResult &result = *resultOpt;

  bool hasReload = result.reloadInterval.has_value();

  // Create IR operations for spill and reload buffers
  Value originalMemref = result.spilledMemref;
  auto type = llvm::cast<MemRefType>(originalMemref.getType());
  Location loc = originalMemref.getLoc();
  OpBuilder builder(cur.memref.getDefiningOp());

  // Spill: alloc + copy original -> spill buffer, placed before the new alloc
  // that triggered the spill.
  Value spillMemref = emitStage(builder, loc, type, originalMemref,
                                cur.memref.getDefiningOp()).memref;

  // Reload: alloc + copy spill -> reload buffer (placed at first use of reload)
  Value reloadMemref;
  if (hasReload && !result.reloadInterval->uses.empty()) {
    Operation *reloadPoint = result.reloadInterval->uses.front().op;
    reloadMemref =
        emitStage(builder, loc, type, spillMemref, reloadPoint).memref;
    memref::DeallocOp::create(builder, loc, spillMemref);

    // Rewrite uses of original memref to use the reload memref
    for (const auto &use : result.reloadInterval->uses) {
      use.op->replaceUsesOfWith(originalMemref, reloadMemref);
    }
  }

  // Add spill interval to next tier
  LiveInterval spillInterval(result.spillBuffer.id, result.spillBuffer.size,
                             result.spillBuffer.lifespan.lower,
                             result.spillBuffer.lifespan.upper,
                             /*uses=*/{}, spillMemref);
  nextTier.addUnhandled(std::move(spillInterval));
  processTier(nextTierIdx, tiers, solver, memMap);

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

  Buffer buffer(cur.id, Interval(cur.start, cur.end), cur.size,
               tier.alignment);

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
/// `reuseGuard` is the address-reuse padding (in liveness time units) applied
/// uniformly to every tier; see MemoryTier::reuseGuard. `alignment` is the
/// required byte alignment applied uniformly to every buffer.
llvm::StringMap<int64_t> allocate(const AllocationProblem &problem,
                                  llvm::ArrayRef<int64_t> tierCapacities,
                                  int64_t reuseGuard = 0,
                                  int64_t alignment = 1) {

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== ALLOCATION START ===\nIntervals ("
                 << problem.intervals.size() << "):\n";
    for (const auto &iv : problem.intervals)
      llvm::dbgs() << "  id=" << iv.id << " size=" << iv.size << " ["
                   << iv.start << ", " << iv.end << ")\n";
    llvm::dbgs() << "Tiers: ";
    for (auto cap : tierCapacities)
      llvm::dbgs() << cap << " ";
    llvm::dbgs() << "(reuseGuard=" << reuseGuard << ")\n";
  });

  // Create memory tiers (level 0 = fastest/smallest, level N = slowest/largest)
  llvm::SmallVector<MemoryTier, 4> tiers;
  for (auto [i, cap] : llvm::enumerate(tierCapacities)) {
    int64_t level = static_cast<int64_t>(i);
    std::string name = "Tier " + std::to_string(level);
    tiers.emplace_back(std::move(name), cap, level, reuseGuard, alignment);
  }

  // Initialize minimalloc solver
  minimalloc::Solver solver;
  llvm::StringMap<int64_t> memMap;

  // Partition intervals into their requested home tier. Defaults to tier 0;
  // synthetic stage intervals and explicit `eaac.home_tier` directives route
  // here. Clamp out-of-range to the bottom tier.
  for (auto &iv : problem.sortedIntervals()) {
    size_t entry =
        static_cast<size_t>(std::clamp<int64_t>(iv.homeTier, 0,
                                                tiers.size() - 1));
    tiers[entry].addUnhandled(std::move(iv));
  }

  // Process bottom-up so residents land before their stagers reference them.
  for (size_t t = tiers.size(); t-- > 0;)
    processTier(t, tiers, solver, memMap);

  LLVM_DEBUG({
    llvm::dbgs() << "\n=== ALLOCATION COMPLETE ===\nFinal memMap ("
                 << memMap.size() << "):\n";
    for (const auto &kv : memMap)
      llvm::dbgs() << "  " << kv.first() << " -> " << kv.second << "\n";
  });

  return memMap;
}



/// Settings read from the EAAC entry of the module's DLTI spec.
struct EaacTargetSettings {
  SmallVector<int64_t> tierCapacities;
  // Padding in liveness time units for address-reuse spreading; see
  // MemoryTier::reuseGuard. Defaults to 0 (no spreading).
  int64_t reuseGuard = 0;
};

/// Look up EAAC settings on the module's `dlti.target_system_spec`. The spec
/// and the `tier_capacities` entry are mandatory; `reuse_guard` is optional.
static FailureOr<EaacTargetSettings> getEaacTargetSettings(ModuleOp module) {
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

  EaacTargetSettings settings;
  bool sawTierCapacities = false;
  for (DataLayoutEntryInterface entry : (*deviceSpec).getEntries()) {
    auto key = dyn_cast<StringAttr>(entry.getKey());
    if (!key)
      continue;
    if (key.getValue() == "tier_capacities") {
      auto arr = dyn_cast<DenseI64ArrayAttr>(entry.getValue());
      if (!arr)
        return module.emitError("'tier_capacities': expected array<i64>, got ")
               << entry.getValue();
      settings.tierCapacities.assign(arr.asArrayRef().begin(),
                                     arr.asArrayRef().end());
      sawTierCapacities = true;
    } else if (key.getValue() == "reuse_guard") {
      auto intAttr = dyn_cast<IntegerAttr>(entry.getValue());
      if (!intAttr)
        return module.emitError("'reuse_guard': expected i64, got ")
               << entry.getValue();
      settings.reuseGuard = intAttr.getInt();
      if (settings.reuseGuard < 0)
        return module.emitError("'reuse_guard' must be non-negative, got ")
               << settings.reuseGuard;
    }
  }
  if (!sawTierCapacities)
    return module.emitError(
        "'tier_capacities' not found in EAAC device spec");
  return settings;
}

/// One planned staging chain: copies needed to transport a buffer up
/// from its home tier to tier 0 in time for a single consumer.
struct StagingChain {
  size_t residentIdx; // index into the intervals vector
  Operation *useOp; // the consumer being staged for
  // TODO, change this to support n tier
  llvm::SmallVector<size_t, 4> stageIdxs; // intervals, descending tier H-1..0
};

/// Build a single staging chain for one (resident, use) pair, appending the
/// `homeTier` synthetic stage intervals (descending tier H-1 → 0) to
/// `intervals`. `residentId`/`residentSize` are taken by value because we
/// mutate `intervals` here; holding a reference into it across emplace_back
/// would be invalidated on growth.
static StagingChain
makeStagingChain(size_t residentIdx, llvm::StringRef residentId,
                 int64_t residentSize, int64_t homeTier, const MemRefUse &use,
                 llvm::SmallVectorImpl<LiveInterval> &intervals) {
  StagingChain chain{residentIdx, use.op, {}};
  for (int64_t K : llvm::reverse(llvm::seq<int64_t>(0, homeTier))) {
    std::string id = (llvm::Twine(residentId) + "_stage" + llvm::Twine(K) +
                      "@" + llvm::Twine(use.time))
                         .str();
    chain.stageIdxs.push_back(intervals.size());
    intervals.emplace_back(std::move(id), residentSize, use.time,
                           use.time + 1,
                           /*uses=*/llvm::SmallVector<MemRefUse>{},
                           /*memref=*/Value(),
                           /*offset=*/-1, /*homeTier=*/K);
  }
  return chain;
}

/// Synthesize per-use staging intervals for every interval with homeTier > 0.
/// Appends synthetic LiveIntervals to `intervals` and returns one StagingChain
/// per use. IR is *not* mutated here — that happens after allocation in
/// emitStagingChains() once offsets are known.
static llvm::SmallVector<StagingChain>
planStagingChains(llvm::SmallVectorImpl<LiveInterval> &intervals) {
  llvm::SmallVector<StagingChain> chains;
  // Snapshot: only the original intervals get chains; the synthetic stages we
  // append below would otherwise be visited recursively.
  const size_t numOriginal = intervals.size();
  for (size_t i : llvm::seq<size_t>(0, numOriginal)) {
    // Copy the resident's fields by value: `intervals` is about to be appended
    // to inside makeStagingChain, which can move the resident element.
    const int64_t homeTier = intervals[i].homeTier;
    if (homeTier <= 0)
      continue;
    const std::string id = intervals[i].id;
    const int64_t size = intervals[i].size;
    const llvm::SmallVector<MemRefUse> uses = intervals[i].uses;
    for (const MemRefUse &use : uses)
      chains.push_back(makeStagingChain(i, id, size, homeTier, use, intervals));
  }
  return chains;
}

/// Materialize the planned staging chains in IR: insert alloc+copy pairs in
/// tier order H-1 → 0 right before each consumer, tag each alloc with its
/// `eaac.tier` / `eaac.offset`, and rewrite the consumer to read from the
/// tier-0 stage memref.
static void emitStagingChains(llvm::ArrayRef<StagingChain> chains,
                              llvm::ArrayRef<LiveInterval> intervals,
                              const llvm::StringMap<int64_t> &memMap) {
  for (const auto &chain : chains) {
    const LiveInterval &resident = intervals[chain.residentIdx];
    Value residentMemref = resident.memref;
    if (!residentMemref)
      continue;

    auto type = llvm::cast<MemRefType>(residentMemref.getType());
    Location loc = residentMemref.getLoc();
    OpBuilder builder(chain.useOp);

    Value src = residentMemref;
    for (size_t stageIdx : chain.stageIdxs) {
      const LiveInterval &stage = intervals[stageIdx];
      StagePoint sp = emitStage(builder, loc, type, src, chain.useOp);
      Operation *allocOp = sp.memref.getDefiningOp();
      auto it = memMap.find(stage.id);
      int64_t offset = it != memMap.end() ? it->second : 0;
      allocOp->setAttr("eaac.tier",
                       builder.getI64IntegerAttr(stage.homeTier));
      allocOp->setAttr("eaac.offset", builder.getI64IntegerAttr(offset));
      if (auto alloc = mlir::dyn_cast<memref::AllocOp>(allocOp)) {
        auto oldType = alloc.getType();
        auto memSpace = eaac::MemSpaceAttr::get(
            allocOp->getContext(), stage.homeTier, offset);
        alloc.getResult().setType(MemRefType::get(
            oldType.getShape(), oldType.getElementType(),
            oldType.getLayout(), memSpace));
      }
      src = sp.memref;
    }
    // Point the consumer at the tier-0 stage in place of the resident.
    chain.useOp->replaceUsesOfWith(residentMemref, src);
  }
}

class LocalStagingPass
    : public impl::LocalStagingBase<LocalStagingPass> {
public:
  using LocalStagingBase::LocalStagingBase;

  void runOnOperation() override {
    FailureOr<EaacTargetSettings> settings =
        getEaacTargetSettings(getOperation());
    if (failed(settings))
      return signalPassFailure();
    int64_t numTiers = static_cast<int64_t>(settings->tierCapacities.size());

    // Get lifetime analysis from the AnalysisManager
    auto &livenessAnalysis = getAnalysis<MemRefLivenessAnalysis>();
    const auto &lifetimeInfos = livenessAnalysis.getIntervals();

    // Mutable working copy: we'll append synthetic staging intervals.
    llvm::SmallVector<LiveInterval> intervals(lifetimeInfos.begin(),
                                              lifetimeInfos.end());

    // Apply kind-based default policy on top of any explicit eaac.home_tier:
    // memref.get_global → bottom tier so constants don't compete for tier 0.
    for (auto &iv : intervals) {
      if (iv.homeTier != 0)
        continue;
      Operation *defOp = iv.memref ? iv.memref.getDefiningOp() : nullptr;
      if (defOp && isa<memref::GetGlobalOp>(defOp))
        iv.homeTier = numTiers - 1;
    }

    // Plan the per-use staging chains (appends synthetic intervals).
    auto chains = planStagingChains(intervals);

    AllocationProblem problem;
    for (const auto &interval : intervals) {
      problem.add(interval);
    }

    int64_t alignment = getAlignment(getOperation()).value_or(1);
    auto map = allocate(problem, settings->tierCapacities,
                        settings->reuseGuard, alignment);

    // Emit IR for the planned chains using the offsets the allocator produced.
    emitStagingChains(chains, intervals, map);
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLocalStagingPass() {
  return std::make_unique<LocalStagingPass>();
}

} // namespace eaac
} // namespace mlir
