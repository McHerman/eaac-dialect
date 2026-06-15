//===- CorrectBroadcast.cpp - TODO ---------------------------------------===//
//
// TODO: pass description.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "correct-broadcast"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_CORRECTBROADCAST
#include "eaac/Passes.h.inc"

namespace {

class CorrectBroadcastPass
    : public impl::CorrectBroadcastBase<CorrectBroadcastPass> {
public:
  using CorrectBroadcastBase::CorrectBroadcastBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) { processFunction(funcOp); });
  }

private:


  using ChainPair = std::pair<eaac::RequireOp, async::ExecuteOp>;

  void processFunction(func::FuncOp funcOp) {

    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });

    // Phase 1: collect requires marked for replacement. Don't mutate the IR
    // here — erasing inside a walk can invalidate the walker's cursor.
    //llvm::SmallVector<RequireOp> toErase;
    llvm::SmallVector<ChainPair> toChainAndErase; 

    funcOp.walk([&](async::ExecuteOp executeOp) {
      executeOp.getBody()->walk([&](RequireOp requireOp) {
        Value token = requireOp.getToken();

        auto users = llvm::to_vector(token.getUsers());
        llvm::sort(users, [&](Operation *a, Operation *b) {
          return opTime.lookup(a) < opTime.lookup(b);
        });

        if (users.size() < 2)
          return;                                  // sole user → keep
        if (users.front() == requireOp.getOperation())
          return;                                  // survivor → keep

        auto survivor = cast<RequireOp>(users.front());
        auto survivorExec = survivor->getParentOfType<async::ExecuteOp>();
        toChainAndErase.emplace_back(requireOp, survivorExec);
      });
    });

    // Phase 2: mutate. Safe to insert/erase here — walks are done.
    for (auto [requireOp, survivorExec] : toChainAndErase) {



      LLVM_DEBUG({
        llvm::dbgs() << "[correct-broadcast] demoting require @t="
                     << opTime.lookup(requireOp) << " to chain in @"
                     << funcOp.getSymName() << "\n"
                     << "  require:        " << *requireOp << "\n"
                     << "  survivor exec:  @t="
                     << opTime.lookup(survivorExec) << "\n"
                     << "  chain predecessor token: "
                     << survivorExec.getToken() << "\n";
      });


      // Insert eaac.chain at the require's position, referencing the
      // surviving require's producing async.execute token.
      OpBuilder builder(requireOp);
      ChainOp::create(builder, requireOp.getLoc(), survivorExec.getToken());

      // Forward downstream uses onto the underlying memref so the verifier
      // stays happy after erase.
      requireOp.getResult().replaceAllUsesWith(requireOp.getMemref());
      requireOp.erase();
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createCorrectBroadcastPass() {
  return std::make_unique<CorrectBroadcastPass>();
}

} // namespace eaac
} // namespace mlir
