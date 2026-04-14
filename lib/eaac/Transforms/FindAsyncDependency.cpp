//===- FindAsyncDependency.cpp - Wire async tokens between regions -------===//
//
// Pass to find and wire async.token dependencies between async.execute regions
// based on memref SSA use-def chains. Since all memrefs are distinct SSA
// values from memref.alloc, there is no aliasing to worry about.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "find-async-dependency"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_FINDASYNCDEPENDENCY
#include "eaac/Passes.h.inc"

namespace {


bool writesTo(Operation *op, Value memref) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    // Effect bound to specific value — match directly.
    if (effect.getValue() == memref)
      return true;
    // Op-level effect (no specific value) — fall back to operand check.
    if (!effect.getValue() && llvm::is_contained(op->getOperands(), memref))
      return true;
  }
  return false;
}

class FindAsyncDependencyPass
    : public impl::FindAsyncDependencyBase<FindAsyncDependencyPass> {
public:
  using FindAsyncDependencyBase::FindAsyncDependencyBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:
  void processFunction(func::FuncOp funcOp) {
    // Collect all async.execute ops in program order.
    llvm::SmallVector<async::ExecuteOp> executeOps;
    funcOp.walk([&](async::ExecuteOp op) { executeOps.push_back(op); });

    // Map each async.execute to its program-order index.
    llvm::DenseMap<Operation *, int64_t> execIndex;
    for (auto [i, op] : llvm::enumerate(executeOps))
      execIndex[op] = i;

    // Track which async.execute ops come before the current one.
    llvm::DenseSet<async::ExecuteOp> preceding;

    for (auto executeOp : executeOps) {
      // For each memref operand, find the latest preceding writer.
      llvm::DenseMap<Value, async::ExecuteOp> latestProducer;

      executeOp.getBody()->walk([&](Operation *innerOp) {
        for (Value memref : innerOp->getOperands()) {
          if (!isa<MemRefType>(memref.getType()))
            continue;

          for (Operation *user : memref.getUsers()) {
            if (user == innerOp)
              continue;
            // Checks if the op is a consumer or producer
            if (!writesTo(user, memref))
              continue;

            auto producerExec = user->getParentOfType<async::ExecuteOp>();
            if (!producerExec || producerExec == executeOp ||
                !preceding.contains(producerExec))
              continue;

            auto it = latestProducer.find(memref);
            if (it == latestProducer.end() ||
                execIndex[producerExec] > execIndex[it->second])
              latestProducer[memref] = producerExec;
          }
        }
      });

      // Collect unique deps from the latest producers.
      llvm::DenseSet<async::ExecuteOp> deps;
      for (auto &[memref, producer] : latestProducer)
        deps.insert(producer);

      if (deps.empty()) {
        preceding.insert(executeOp);
        continue;
      }

      // Rebuild async.execute after the current position with dependency tokens.
      llvm::SmallVector<Value> newDeps(executeOp.getDependencies());
      for (auto dep : deps)
        newDeps.push_back(dep.getToken());

      OpBuilder builder(executeOp->getBlock(), ++Block::iterator(executeOp));
      auto newOp = async::ExecuteOp::create(
          builder, executeOp.getLoc(),
          /*resultTypes=*/TypeRange{},
          /*dependencies=*/newDeps,
          /*operands=*/ValueRange{});

      // Move body.
      newOp.getBody()->getOperations().clear();
      newOp.getBody()->getOperations().splice(
          newOp.getBody()->end(),
          executeOp.getBody()->getOperations());

      executeOp.getToken().replaceAllUsesWith(newOp.getToken());
      execIndex[newOp] = execIndex[executeOp];
      preceding.insert(newOp);
      executeOp.erase();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createFindAsyncDependencyPass() {
  return std::make_unique<FindAsyncDependencyPass>();
}

} // namespace eaac
} // namespace mlir
