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

    // Collect all but the first require in order of execution when encountering broadcast.
    llvm::SmallVector<ChainPair> toRewire;

    funcOp.walk([&](async::ExecuteOp executeOp) {
      executeOp.getBody()->walk([&](RequireOp requireOp) {
        Value token = requireOp.getToken();

        auto users = llvm::to_vector(token.getUsers());
        llvm::sort(users, [&](Operation *a, Operation *b) {
          return opTime.lookup(a) < opTime.lookup(b);
        });

        if (users.size() < 2) // A given operation only has a single consumer, and is therefore not a broadcast.
          return;                                  // sole user → keep
        if (users.front() == requireOp.getOperation()) // The given op is the first of all broadcast consumers and will therefor be first chain link
          return;                                  // survivor → keep

        auto survivor = cast<RequireOp>(users.front());
        toRewire.emplace_back(requireOp, survivor);
      });
    });

    // Rewrite
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


      // Create new execute region, this is essentially just to trick later pipelines into creating a semaphore.
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

      // Change out the original token dependency with the new generated chained sem.
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
