//===- FindAsyncDependency.cpp - Wire async tokens between regions -------===//
//
// Pass to find and wire async.token dependencies between async.execute
// regions based on memref SSA use-def chains.
//
// Invariant: every memref has at most one writer across the function.
// Under that invariant the dependency rule is just: for each memref read by
// an async.execute, depend on the (unique) async.execute that wrote it.
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

// Returns true iff `op` declares a value-bound Write effect on `memref`.
// We deliberately ignore op-level Write effects with no bound value: in this
// dialect those are used to model writes to external state (e.g. eaac.store)
// to prevent DCE, not writes to any operand. Treating them as operand writes
// would falsely attribute a writer to read-only operands.
bool writesTo(Operation *op, Value memref) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return false;
  llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
  effectOp.getEffects(effects);
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    if (effect.getValue() == memref)
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
  // Populate `writer` with memref → producing async.execute. Asserts the
  // single-writer invariant: a memref must not be written by two different
  // async.executes.
  static void buildWriterMap(
      llvm::ArrayRef<async::ExecuteOp> executeOps,
      llvm::DenseMap<Value, async::ExecuteOp> &writer) {
    for (auto exec : executeOps) {
      exec.getBody()->walk([&](Operation *innerOp) {
        for (Value operand : innerOp->getOperands()) {
          if (!isa<MemRefType>(operand.getType()))
            continue;
          if (!writesTo(innerOp, operand))
            continue;
          auto [it, inserted] = writer.try_emplace(operand, exec);
          assert((inserted || it->second == exec) &&
                 "memref written by more than one async.execute");
          (void)it;
          (void)inserted;
        }
      });
    }
  }

  // Collect the set of async.executes that produced any memref read inside
  // `exec`, excluding `exec` itself.
  static llvm::DenseSet<async::ExecuteOp> collectDeps(
      async::ExecuteOp exec,
      const llvm::DenseMap<Value, async::ExecuteOp> &writer) {
    llvm::DenseSet<async::ExecuteOp> deps;
    exec.getBody()->walk([&](Operation *innerOp) {
      for (Value operand : innerOp->getOperands()) {
        if (!isa<MemRefType>(operand.getType()))
          continue;
        auto it = writer.find(operand);
        if (it == writer.end() || it->second == exec)
          continue;
        deps.insert(it->second);
      }
    });
    return deps;
  }

  // Replace `exec` with an equivalent async.execute that additionally
  // depends on `deps`. Returns the new op.
  static async::ExecuteOp rewriteWithDeps(
      async::ExecuteOp exec, const llvm::DenseSet<async::ExecuteOp> &deps) {
    llvm::SmallVector<Value> newDeps(exec.getDependencies());
    for (auto dep : deps)
      newDeps.push_back(dep.getToken());

    OpBuilder builder(exec->getBlock(), ++Block::iterator(exec));
    auto newOp = async::ExecuteOp::create(
        builder, exec.getLoc(),
        /*resultTypes=*/TypeRange{},
        /*dependencies=*/newDeps,
        /*operands=*/ValueRange{});

    newOp.getBody()->getOperations().clear();
    newOp.getBody()->getOperations().splice(
        newOp.getBody()->end(), exec.getBody()->getOperations());

    exec.getToken().replaceAllUsesWith(newOp.getToken());
    exec.erase();
    return newOp;
  }

  void processFunction(func::FuncOp funcOp) {
    llvm::SmallVector<async::ExecuteOp> executeOps;
    funcOp.walk([&](async::ExecuteOp op) { executeOps.push_back(op); });

    llvm::DenseMap<Value, async::ExecuteOp> writer;
    buildWriterMap(executeOps, writer);

    for (auto exec : executeOps) {
      auto deps = collectDeps(exec, writer);
      if (deps.empty())
        continue;
      async::ExecuteOp newOp = rewriteWithDeps(exec, deps);
      // Keep the writer map valid: any memref previously attributed to
      // `exec` is now produced by `newOp`.
      for (auto &entry : writer)
        if (entry.second == exec)
          entry.second = newOp;
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createFindAsyncDependencyPass() {
  return std::make_unique<FindAsyncDependencyPass>();
}

} // namespace eaac
} // namespace mlir
