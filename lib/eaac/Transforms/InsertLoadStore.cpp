//===- InsertLoadStore.cpp - Replace func args/returns with load/store ----===//
//
// Flattens the module hierarchy and replaces main function parameters with
// eaac.load ops and return values with eaac.store ops.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "eaac-insert-load-store"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_INSERTLOADSTORE
#include "eaac/Passes.h.inc"

namespace {

class InsertLoadStorePass
    : public impl::InsertLoadStoreBase<InsertLoadStorePass> {
public:
  using InsertLoadStoreBase::InsertLoadStoreBase;

  void runOnOperation() override {


    // 1. For each block argument, insert eaac.load and replace uses
    // 2. For each func.return operand, insert eaac.store
    // 3. Move body ops into the module
    // 4. Erase the func


    ModuleOp moduleOp = getOperation();

    // Should only be the main function
    SmallVector<func::FuncOp> functions;
    // Collect first to avoid modifying while iterating.
    moduleOp.walk([&](func::FuncOp f) { functions.push_back(f);});

    for(auto funcOp : functions){

      OpBuilder builder(&funcOp.getBody().front(), funcOp.getBody().front().begin());

      for (BlockArgument arg : funcOp.getArguments()) {
        auto memrefType = dyn_cast<MemRefType>(arg.getType());
        if (!memrefType)
          continue;   

        // Create a contiguous memref type (drop dynamic strides/offset from bufferization)
        auto allocType = MemRefType::get(memrefType.getShape(),memrefType.getElementType());

        auto alloc = memref::AllocOp::create(builder, funcOp.getLoc(), allocType);

        eaac::LoadOp::create(builder, funcOp.getLoc(), alloc);
        arg.replaceAllUsesWith(alloc);
      }    

      // Erase block arguments (reverse order to keep indices valid)
      for (unsigned i = funcOp.getNumArguments(); i-- > 0;)
        funcOp.getBody().front().eraseArgument(i);

      // Find the return op
      auto returnOp = cast<func::ReturnOp>(funcOp.getBody().front().getTerminator());

      // Insert stores before the return
      builder.setInsertionPoint(returnOp);
      for (Value operand : returnOp.getOperands()) {
        eaac::StoreOp::create(builder, funcOp.getLoc(), operand);
      }

      // Drop all return operands (keep the terminator)
      returnOp->eraseOperands(0, returnOp.getNumOperands());

      // Update function type to have no results
      funcOp.setFunctionType(FunctionType::get(funcOp.getContext(), {}, {})); 
    }


      /* 
      Block *moduleBlock = moduleOp.getBody();
      Block *funcBlock = &funcOp.getBody().front();

      moduleBlock->getOperations().splice(
        Block::iterator(funcOp),  // insert before the funcOp itself
        funcBlock->getOperations());

      funcOp.erase();
      */
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createInsertLoadStorePass() {
  return std::make_unique<InsertLoadStorePass>();
}

} // namespace eaac
} // namespace mlir
