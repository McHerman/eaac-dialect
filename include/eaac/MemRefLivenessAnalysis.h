//===- MemRefLivenessAnalysis.h - MemRef lifetime analysis -------*- C++ -*-===//
//
// Analysis that computes live intervals for all memref allocations.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_MEMREFLIVENESSANALYSIS_H
#define EAAC_MEMREFLIVENESSANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"

#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace eaac {

/// Stores information about a single use of a memref.
struct MemRefUse {
  Operation *op;
  int64_t time;
};

/// A live range with use positions for a memref allocation.
struct LiveInterval {
  std::string id;
  int64_t size;
  int64_t start;
  int64_t end;
  llvm::SmallVector<MemRefUse> uses;
  int64_t offset;
  std::optional<int64_t> reloadFromTier;
  std::optional<int64_t> reloadFromOffset;

  /// Reference to the memref SSA value this interval tracks.
  Value memref;


  LiveInterval()
      : size(0), start(0), end(0), offset(-1), reloadFromTier(std::nullopt),
        reloadFromOffset(std::nullopt) {}

  LiveInterval(std::string id, int64_t size, int64_t start, int64_t end,
               llvm::SmallVector<MemRefUse> uses = {}, Value memref = Value(),
               int64_t offset = -1)
      : id(std::move(id)), size(size), start(start), end(end),
        uses(std::move(uses)), offset(offset), reloadFromTier(std::nullopt),
        reloadFromOffset(std::nullopt), memref(memref) {}

  /// Comparison operator for sorting by start position.
  bool operator<(const LiveInterval &other) const { return start < other.start; }
};

/// Analysis that computes live intervals for all memref allocations in a module.
///
/// Usage from a pass:
///   auto &analysis = getAnalysis<MemRefLivenessAnalysis>();
///   const auto &intervals = analysis.getIntervals();
class MemRefLivenessAnalysis {
public:
  MemRefLivenessAnalysis(Operation *op);

  /// Get the computed live intervals.
  const llvm::SmallVector<LiveInterval> &getIntervals() const {
    return intervals;
  }

  /// Invalidate when any analysis is not preserved.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<MemRefLivenessAnalysis>();
  }

private:
  llvm::SmallVector<LiveInterval> intervals;
};

} // namespace eaac
} // namespace mlir

#endif // EAAC_MEMREFLIVENESSANALYSIS_H
