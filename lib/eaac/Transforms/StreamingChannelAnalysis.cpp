//===- StreamingChannelAnalysis.cpp - Streaming-FU channel grouping ------===//
//
// Walks sem_alloc ops in program order; for each one, looks at the producing
// eaac.execute body (identified via the SemAcquireOp that lives inside it)
// and asks the per-op classifier whether its inner op belongs to a streaming
// channel. Matching semaphores get assigned per-channel queue indices.
//
// Adding a new streaming FU: extend `getChannel(Operation *)` below. No
// allocator change required — AssignSemaphoreAddresses iterates whatever
// channels this analysis surfaces and reserves a per-channel ring for each.
//
//===----------------------------------------------------------------------===//

#include "eaac/StreamingChannelAnalysis.h"

#include "eaac/Dialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "streaming-channel-analysis"

namespace mlir {
namespace eaac {

namespace {

/// Channel registration 
std::optional<ChannelSpec> getChannel(Operation *op) {
  // Accept either pre- or post-LinalgToEaac conversion form.
  if (isa<linalg::MatmulOp, MatmulOp>(op))
    return ChannelSpec{"matmul", /*depth=*/2};
  return std::nullopt;
}

/// Find the channel of the eaac.execute that uses `semaphore`.
std::pair<ExecuteOp, std::optional<ChannelSpec>>
findProducer(Value semaphore) {
  for (Operation *user : semaphore.getUsers()) {
    if (!isa<SemAcquireOp>(user))
      continue;
    auto exec = user->getParentOfType<ExecuteOp>();
    if (!exec)
      continue;
    std::optional<ChannelSpec> spec;
    exec.getBody().walk([&](Operation *innerOp) {
      if (auto s = getChannel(innerOp)) {
        spec = s;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return {exec, spec};
  }
  return {nullptr, std::nullopt};
}

} // namespace

StreamingChannelAnalysis::StreamingChannelAnalysis(Operation *op) {
  auto module = cast<ModuleOp>(op);
  // Per-channel queue counters.
  llvm::StringMap<ChannelSpec> seen;

  module.walk([&](func::FuncOp funcOp) {
    llvm::StringMap<int64_t> counters;
    funcOp.walk([&](SemAllocOp allocOp) {
      Value sem = allocOp.getSemaphore();
      auto [exec, spec] = findProducer(sem);
      if (!spec)
        return;
      int64_t idx = counters[spec->name]++;
      ChannelAssignment a{spec->name, spec->depth, idx};
      assignments[sem] = a;
      if (exec)
        assignmentsByExecute[exec] = a;
      seen.try_emplace(spec->name, *spec);
      LLVM_DEBUG(llvm::dbgs() << "[streaming-channel] " << sem
                              << " → " << spec->name << "[" << idx << "/"
                              << spec->depth << "]\n");
    });
  });

  for (auto &kv : seen)
    channels.push_back(kv.second);
}

} // namespace eaac
} // namespace mlir
