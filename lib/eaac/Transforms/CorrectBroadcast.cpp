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


  using ChainPair = std::pair<eaac::RequireOp, eaac::RequireOp>;

  void processFunction(func::FuncOp funcOp) {

    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });

    // Phase 1: collect non-survivor requires. Don't mutate the IR here —
    // creating/erasing inside a walk can invalidate the walker's cursor.
    llvm::SmallVector<ChainPair> toRewire;

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
        toRewire.emplace_back(requireOp, survivor);
      });
    });

    // Phase 2: mutate. Safe to create/rewire here — walks are done.
    for (auto [requireOp, survivor] : toRewire) {
      Value broadcastToken = requireOp.getToken();
      auto consumerExec = requireOp->getParentOfType<async::ExecuteOp>();
      Location loc = requireOp.getLoc();

      LLVM_DEBUG({
        llvm::dbgs() << "[correct-broadcast] threading require @t="
                     << opTime.lookup(requireOp) << " through chain exec in @"
                     << funcOp.getSymName() << "\n"
                     << "  require:   " << *requireOp << "\n"
                     << "  survivor:  " << *survivor << "\n";
      });

      // Build a new async.execute just before the consumer exec. It depends
      // on the broadcast token and holds a single eaac.chain, so its
      // semaphore pair is the one the consumer ends up waiting on instead of
      // the shared broadcast token.
      OpBuilder builder(consumerExec);
      auto chainExec = async::ExecuteOp::create(
          builder, loc,
          /*resultTypes=*/TypeRange{},
          /*dependencies=*/ValueRange{broadcastToken},
          /*operands=*/ValueRange{});

      Block *body = chainExec.getBody();
      if (!body->empty())
        body->back().erase();             // strip auto-generated yield
      OpBuilder bodyBuilder(body, body->end());
      ChainOp::create(bodyBuilder, loc, broadcastToken,
                      /*is_broadcast=*/true);
      async::YieldOp::create(bodyBuilder, loc, ValueRange{});

      // Rewire: the consumer's broadcast-token dependency becomes a
      // dependency on the new chain exec, and the require's token operand
      // points at the new token too.
      Value newToken = chainExec.getToken();
      for (auto [i, dep] : llvm::enumerate(consumerExec.getDependencies())) {
        if (dep == broadcastToken) {
          consumerExec->setOperand(i, newToken);
          break;
        }
      }
      requireOp->setOperand(0, newToken);
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createCorrectBroadcastPass() {
  return std::make_unique<CorrectBroadcastPass>();
}

} // namespace eaac
} // namespace mlir
