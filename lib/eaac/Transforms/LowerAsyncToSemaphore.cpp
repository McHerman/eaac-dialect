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

    auto semReq = SemRequireOp::create(rewriter, op.getLoc(),
                                       op.getMemref().getType(),
                                       it->second, op.getMemref());
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

      // Apply rewrite patterns.
      RewritePatternSet patterns(&getContext());
      patterns.add<RequireToSemRequire>(&getContext(), tokenToSem);
      patterns.add<AsyncExecuteToEaacExecute>(&getContext());

      if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
        signalPassFailure();

      // Insert sem_dealloc before function return.
      insertSemDeallocs(funcOp, tokenToSem);
    });
  }

private:
  /// Compute the size in bytes of a statically-shaped memref.
  int64_t getMemRefSizeInBytes(MemRefType type) {
    int64_t numElements = 1;
    for (int64_t dim : type.getShape())
      numElements *= dim;
    int64_t elementBits = type.getElementTypeBitWidth();
    return numElements * elementBits / 8;
  }

  /// Walk all async.execute ops, find token→require→memref edges,
  /// insert sem_alloc before each producer, populate tokenToSem.
  void buildTokenToSemMap(func::FuncOp funcOp,
                          llvm::DenseMap<Value, Value> &tokenToSem) {
    OpBuilder builder(funcOp);
    auto loc = funcOp.getLoc();
    auto semTy = SemaphoreType::get(funcOp.getContext());

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
            SemAllocOp::create(builder, loc, semTy, emptyCount, fullCount);
        tokenToSem[token] = sem.getSemaphore();

        // Insert sem_acquire at the start of the producer's body.
        builder.setInsertionPointToStart(producerExec.getBody());
        SemAcquireOp::create(builder, producerExec.getLoc(), memrefTy,
                             sem.getSemaphore(), memref);
      }
    }
  }

  /// Insert sem_dealloc after the consumer eaac.execute that uses the semaphore.
  void insertSemDeallocs(func::FuncOp funcOp,
                         llvm::DenseMap<Value, Value> &tokenToSem) {
    OpBuilder builder(funcOp);
    for (auto &[token, sem] : tokenToSem) {
      for (Operation *user : sem.getUsers()) {
        auto parentExec = user->getParentOfType<ExecuteOp>();
        if (!parentExec)
          continue;
        builder.setInsertionPointAfter(parentExec);
        SemDeallocOp::create(builder, parentExec.getLoc(), sem);
        break;
      }
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createLowerAsyncToSemaphorePass() {
  return std::make_unique<LowerAsyncToSemaphorePass>();
}

} // namespace eaac
} // namespace mlir
