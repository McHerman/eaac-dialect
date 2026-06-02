// Chcekcs 

#ifndef EAAC_STREAMINGCHANNELANALYSIS_H
#define EAAC_STREAMINGCHANNELANALYSIS_H

#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace eaac {

/// Identity + drain pipeline depth of a streaming channel.
struct ChannelSpec {
  llvm::StringRef name;
  int64_t depth; // K (>= 1)
};

/// What the analysis records per signal-semaphore: which channel it's on and
/// its queue position within that channel.
struct ChannelAssignment {
  llvm::StringRef name;
  int64_t depth;
  int64_t index;
};

class StreamingChannelAnalysis {
public:
  static llvm::DenseMap<Operation *, int64_t>
  buildOpTimeMap(func::FuncOp funcOp) {
    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });
    return opTime;
  }

  StreamingChannelAnalysis(Operation *op);

  /// Checks if a give semaphore belongs to a channel
  std::optional<ChannelAssignment> getAssignment(Value semaphore) const {
    auto it = assignments.find(semaphore);
    if (it == assignments.end())
      return std::nullopt;
    return it->second;
  }

  /// Same lookup but keyed by the producing eaac.execute op. Convenient when
  /// the caller has an `Operation *` in hand (e.g. a sem_require's enclosing
  /// execute) and would otherwise have to fish for its output semaphore.
  std::optional<ChannelAssignment> getAssignment(Operation *execute) const {
    auto it = assignmentsByExecute.find(execute);
    if (it == assignmentsByExecute.end())
      return std::nullopt;
    return it->second;
  }

  /// Channels observed across the analyzed module. AssignSemaphoreAddresses
  /// iterates these to know which rings to reserve.
  llvm::ArrayRef<ChannelSpec> getChannels() const { return channels; }

  /// Invalidate when any analysis is not preserved.
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa) {
    return !pa.isPreserved<StreamingChannelAnalysis>();
  }

private:
  llvm::DenseMap<Value, ChannelAssignment> assignments;
  llvm::DenseMap<Operation *, ChannelAssignment> assignmentsByExecute;
  llvm::SmallVector<ChannelSpec> channels;
};

} // namespace eaac
} // namespace mlir

#endif // EAAC_STREAMINGCHANNELANALYSIS_H
