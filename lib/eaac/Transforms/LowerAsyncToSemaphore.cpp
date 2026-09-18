//===- LowerAsyncToSemaphore.cpp - Async tokens to hardware semaphores ---===//
//
// Converts async.execute regions and token dependencies into hardware
// semaphore operations (sem_alloc, sem_dealloc, semaphore args on ops).
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-async-to-semaphore"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_LOWERASYNCTOSEMAPHORE
#include "eaac/Passes.h.inc"

namespace {

/// Compute the size in bytes of a statically-shaped memref.
static int64_t getMemRefSizeInBytes(MemRefType type) {
  int64_t n = 1;
  for (int64_t dim : type.getShape())
    n *= dim;
  return n * type.getElementTypeBitWidth() / 8;
}

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Replace eaac.require with eaac.sem_require using the token→sem map.
struct RequireToSemRequire : OpRewritePattern<RequireOp> {
  llvm::DenseMap<Value, Value> &tokenToSem;

  RequireToSemRequire(MLIRContext *ctx, llvm::DenseMap<Value, Value> &map)
      : OpRewritePattern(ctx), tokenToSem(map) {}

  LogicalResult matchAndRewrite(RequireOp op,
                                PatternRewriter &rewriter) const override {
    auto it = tokenToSem.find(op.getToken());
    if (it == tokenToSem.end())
      return failure();

    auto memrefTy = cast<MemRefType>(op.getMemref().getType());
    auto semReq = SemRequireOp::create(rewriter, op.getLoc(), memrefTy,
                                       it->second, op.getMemref(),
                                       /*chains_from=*/ValueRange{});
    semReq.setStepSize(getMemRefSizeInBytes(memrefTy));
    rewriter.replaceOp(op, semReq.getResult());
    return success();
  }
};

/// Replace async.execute with eaac.execute, inlining the body.
struct AsyncExecuteToEaacExecute : OpRewritePattern<async::ExecuteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(async::ExecuteOp op,
                                PatternRewriter &rewriter) const override {
    auto newExec = ExecuteOp::create(rewriter, op.getLoc());

    // Move body from async.execute into eaac.execute.
    newExec.getBody().getBlocks().clear();
    newExec.getBody().getBlocks().splice(newExec.getBody().end(),
                                         op.getBodyRegion().getBlocks());

    // Erase async.yield terminators.
    llvm::SmallVector<async::YieldOp> yields;
    newExec.getBody().walk([&](async::YieldOp y) { yields.push_back(y); });
    for (auto y : yields)
      y.erase();

    // Drop token uses and erase the original.
    op.getToken().dropAllUses();
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class LowerAsyncToSemaphorePass
    : public impl::LowerAsyncToSemaphoreBase<LowerAsyncToSemaphorePass> {
public:
  using LowerAsyncToSemaphoreBase::LowerAsyncToSemaphoreBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk([&](func::FuncOp funcOp) {
      // Pre-pass: build token → semaphore map.
      llvm::DenseMap<Value, Value> tokenToSem;
      buildTokenToSemMap(funcOp, tokenToSem);

      // Stage 1: convert eaac.require → eaac.sem_require. After this,
      // tokens are only referenced by async.execute.dependencies, which
      // simplifies the broadcast cleanup in chainSem.
      {
        RewritePatternSet patterns(&getContext());
        patterns.add<RequireToSemRequire>(&getContext(), tokenToSem);
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
          signalPassFailure();
      }

      // Stage 2: resolve eaac.chain ops; tear down broadcast chain execs.
      chainSem(funcOp, tokenToSem);

      // Stage 3: convert async.execute → eaac.execute and drop tokens.
      {
        RewritePatternSet patterns(&getContext());
        patterns.add<AsyncExecuteToEaacExecute>(&getContext());
        if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
          signalPassFailure();
      }

      // Insert sem_dealloc before function return.
      insertSemDeallocs(funcOp, tokenToSem);
    });
  }

private:
  /// Walk all async.execute ops, find token→require→memref edges,
  /// insert sem_alloc before each producer, populate tokenToSem.
  void buildTokenToSemMap(func::FuncOp funcOp,
                          llvm::DenseMap<Value, Value> &tokenToSem) {
    OpBuilder builder(funcOp);
    auto loc = funcOp.getLoc();
    int addr = 0;
    int gen = 0;
    auto semTy = SemaphoreType::get(funcOp.getContext(), addr, gen);

    llvm::SmallVector<async::ExecuteOp> executeOps;
    funcOp.walk([&](async::ExecuteOp op) { executeOps.push_back(op); });

    for (auto consumerExec : executeOps) {
      for (Value token : consumerExec.getDependencies()) {
        if (tokenToSem.count(token))
          continue;

        auto producerExec = token.getDefiningOp<async::ExecuteOp>();
        if (!producerExec)
          continue;

        // Find the eaac.require inside the consumer that uses this token.
        RequireOp requireOp = nullptr;
        consumerExec.getBody()->walk([&](RequireOp req) {
          if (req.getToken() == token)
            requireOp = req;
        });
        if (!requireOp)
          continue;

        Value memref = requireOp.getMemref();
        auto memrefTy = cast<MemRefType>(memref.getType());
        int64_t sizeBytes = getMemRefSizeInBytes(memrefTy);

        // Insert sem_alloc before the producer.
        builder.setInsertionPoint(producerExec);
        auto emptyCount =
            arith::ConstantIndexOp::create(builder, loc, sizeBytes);
        auto fullCount = arith::ConstantIndexOp::create(builder, loc, 0);
        auto sem =
            SemAllocOp::create(builder, loc, semTy, emptyCount, fullCount,
                               /*chains_from=*/ValueRange{});
        tokenToSem[token] = sem.getSemaphore();

        // Insert sem_acquire at the start of the producer's body.
        builder.setInsertionPointToStart(producerExec.getBody());
        auto acq = SemAcquireOp::create(builder, producerExec.getLoc(), memrefTy,
                                        sem.getSemaphore(), memref);
        acq.setStepSize(sizeBytes);
      }
    }
  }

  // Append chained semaphores to semaphore allocation
  void chainSem(func::FuncOp funcOp,
                const llvm::DenseMap<Value, Value> &tokenToSem) {
    llvm::SmallVector<eaac::ChainOp> chainOps;
    funcOp.walk([&](eaac::ChainOp op) { chainOps.push_back(op); });

    // Process non-broadcast (alias) chains before broadcast chains.
    llvm::stable_sort(chainOps, [](eaac::ChainOp a, eaac::ChainOp b) {
      return !a.getIsBroadcast() && b.getIsBroadcast();
    });

    for (eaac::ChainOp chainOp : chainOps) {
      bool isBroadcast = chainOp.getIsBroadcast();

      // This is the op which is chained to previous op
      auto producerExec = chainOp->getParentOfType<async::ExecuteOp>();
      if (!producerExec) {
        chainOp.emitWarning(
            "eaac.chain not enclosed by an async.execute; dropping");
        chainOp.erase();
        continue;
      }

      // This is the semaphore of the which the op is chained to
      auto it = tokenToSem.find(chainOp.getPredecessor());
      if (it == tokenToSem.end()) {
        chainOp.emitWarning(
            "eaac.chain predecessor token has no associated semaphore "
            "(the predecessor producer has no required outputs); "
            "anti-aliasing dependency cannot be enforced — verify that "
            "the alias detection pass produced a meaningful predecessor");
        chainOp.erase();
        continue;
      }
      Value predSem = it->second;

      // Find the acquire op
      SemAcquireOp producerAcquire = nullptr;
      producerExec.getBody()->walk([&](SemAcquireOp acq) {
        producerAcquire = acq;
        return WalkResult::interrupt();
      });
      if (!producerAcquire) {
        chainOp.emitWarning(
            "eaac.chain's enclosing async.execute has no sem_acquire "
            "(its output is unused), so the anti-aliasing chain has "
            "nothing to attach to and is being dropped");
        chainOp.erase();
        continue;
      }

      // Acquire semaphore allocation.
      auto producerAlloc =
          producerAcquire.getSemaphore().getDefiningOp<SemAllocOp>();
      if (!producerAlloc) {
        chainOp.emitWarning(
            "eaac.chain producer's sem_acquire is not paired with a "
            "sem_alloc; cannot attach anti-aliasing chain, dropping");
        chainOp.erase();
        continue;
      }

      // Broadcast chains: the chain semaphore starts full (data is already
      // available from the broadcast producer), so we swap empty/full counts
      // and tear down the chain exec entirely.
      if (isBroadcast) {
        Value e = producerAlloc.getEmptyCount();
        Value f = producerAlloc.getFullCount();
        producerAlloc->setOperand(0, f);
        producerAlloc->setOperand(1, e);

        //producerAlloc->insertOperands(producerAlloc->getNumOperands(),
        //                              predSem);

        producerAlloc.setEventMode(EventMode::R);

        // Redirect the remaining dep uses to the broadcast token (those
        // dep operands get dropped by AsyncExecuteToEaacExecute anyway).
        Value chainToken = producerExec.getToken();
        chainToken.replaceAllUsesWith(chainOp.getPredecessor());

        // Removes unessecary artificial async::ExecuteOp only containing chain.
        producerExec.erase();
        //continue;
      }


      // Append predSem to the producer's chains_from.
      producerAlloc->insertOperands(producerAlloc->getNumOperands(), predSem);

      // In the broadcast case, chainOp was already destroyed when
      // producerExec (its parent) was erased above.
      if (!isBroadcast)
        chainOp.erase();
    }
  }




  /// Insert sem_dealloc after the consumer eaac.execute that uses the semaphore.
  void insertSemDeallocs(func::FuncOp funcOp,
                         llvm::DenseMap<Value, Value> &tokenToSem) {
    OpBuilder builder(funcOp);
    // Iterate semaphores in deterministic IR order (walk over sem_allocs)
    // rather than DenseMap hash order, so that sem_deallocs placed after the
    // same ExecuteOp get a stable relative ordering. Downstream passes
    // (e.g. AssignSemaphoreAddresses) depend on this op ordering.
    funcOp.walk([&](SemAllocOp allocOp) {
      Value sem = allocOp.getSemaphore();
      for (Operation *user : sem.getUsers()) {
        auto parentExec = user->getParentOfType<ExecuteOp>();
        if (!parentExec)
          continue;
        builder.setInsertionPointAfter(parentExec);
        SemDeallocOp::create(builder, parentExec.getLoc(), sem);
        break;
      }
    });
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLowerAsyncToSemaphorePass() {
  return std::make_unique<LowerAsyncToSemaphorePass>();
}

} // namespace eaac
} // namespace mlir
