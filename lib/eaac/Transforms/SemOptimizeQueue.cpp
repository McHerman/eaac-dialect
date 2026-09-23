#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <cstdint>

#define DEBUG_TYPE "eaac-sem-optimize-queue"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_SEMOPTIMIZEQUEUE
#include "eaac/Passes.h.inc"

namespace {

class SemOptimizeQueuePass
    : public impl::SemOptimizeQueueBase<SemOptimizeQueuePass> {
public:
  using SemOptimizeQueueBase::SemOptimizeQueueBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    processQueueSerialization(moduleOp);
  }

private:

  struct workStruct{
    eaac::ExecuteOp op;
    SymbolRefAttr producerUnit;
    eaac::ExecuteOp producerOp;
  };

  static std::pair<llvm::SmallVector<mlir::Value>, llvm::SmallVector<eaac::SemAllocOp>> checkChain(eaac::SemAllocOp op) {

    llvm::SmallVector<mlir::Value> chain;
    llvm::SmallVector<eaac::SemAllocOp> chained;

    for(auto chainSemaphore : op.getChainsFrom()) {
      eaac::SemaphoreType semType = dyn_cast<eaac::SemaphoreType>(chainSemaphore.getType());

      if(semType) {
        chain.push_back(chainSemaphore);
      }
    }


    for(auto user : op.getResult().getUsers()) {
      eaac::SemAllocOp alloc = dyn_cast<eaac::SemAllocOp>(user);

      if(alloc) {
        chained.push_back(alloc);
      }
    }
    
    return std::pair(chain, chained);
  }


  static eaac::SemAllocOp findSemAlloc(eaac::ExecuteOp producer) {
    auto acquire = producer.getAcquireOp();

    if(!acquire)
      return nullptr;

    auto semaphore = acquire.getSemaphore();
    auto semAlloc = semaphore.getDefiningOp();

    if(semAlloc) {
      return dyn_cast<eaac::SemAllocOp>(semAlloc);
    } else {
      return nullptr;
    }
  }

  static std::optional<int64_t> findQueueDepth(mlir::Operation *op) {

    auto hwUnit = dyn_cast<eaac::HardwareUnitOpInterface>(op);
    if (!hwUnit) {
      LLVM_DEBUG(llvm::dbgs() << "Op in schedule region has no HardwareUnitOpInterface\n");
      return std::nullopt;
    }   

    auto funitIface = dyn_cast<eaac::FunitTypeInterface>(hwUnit.getFunitType());
    if (!funitIface) {
      LLVM_DEBUG(llvm::dbgs() << "Funit type has no FunitTypeInterface\n");
      return std::nullopt;
    }

    LLVM_DEBUG(llvm::dbgs() << "found hardware op operand\n");

    return funitIface.getQueueDepth();
  }


  static void processQueueSerialization(ModuleOp moduleOp) {
    auto map = buildSchedule(moduleOp);
    mlir::AsmState asmState(moduleOp);

    //llvm::<mlir::Operation*, int64_t> work;
    llvm::SmallVector<workStruct> work;

    moduleOp.walk([&](eaac::ExecuteOp op) {
      llvm::SmallVector<eaac::SemRequireOp> requireOps = op.getRequireOps();

      auto symbol = op.getPayloadOp()->getAttrOfType<SymbolRefAttr>("hw_unit");

      if(requireOps.empty())
        return WalkResult::advance();

      for (eaac::SemRequireOp require : requireOps) {
        if(!require)
          continue;

        eaac::SemAcquireOp producer = require.getProducer();

        if(!producer)
          continue;

        eaac::ExecuteOp producerExecute = dyn_cast<eaac::ExecuteOp>(producer->getParentOp());

        auto producerSymbol =
            producerExecute.getPayloadOp()->getAttrOfType<SymbolRefAttr>("hw_unit");

        if (symbol && producerSymbol && producerSymbol != symbol)
          work.push_back(workStruct(op,producerSymbol,producerExecute));
      }

      return WalkResult::advance();
    });

    llvm::MapVector<Operation *, int64_t> ops;
    int64_t count = 0;
    moduleOp.walk([&](Operation *op) {
      ops[op] = count; 
      count += 1;
    });

    for (workStruct workItem : work) {
      eaac::ExecuteOp consumerOp = workItem.op;

      int64_t targetIndex = ops.find(consumerOp)->second;
      
      if(targetIndex == ops.end()->second) {
        LLVM_DEBUG(llvm::dbgs() << "WorkOp not found: ";
                   consumerOp.getPayloadOp()->print(llvm::dbgs(), asmState);
                   llvm::dbgs() << "\n");
        continue;
      }

      //Operation* lastInstanceOfType = nullptr;
      eaac::ExecuteOp lastInstanceOfType = nullptr;

      for (auto &[key, value] : llvm::reverse(ops)) {
        eaac::ExecuteOp executeOp = dyn_cast<eaac::ExecuteOp>(key);

        if(!executeOp)
          continue;

        auto executeSymbol = executeOp.getPayloadOp()->getAttrOfType<SymbolRefAttr>("hw_unit");

        if(executeSymbol == workItem.producerUnit && executeOp->isBeforeInBlock(consumerOp)) {
          lastInstanceOfType = executeOp;
          LLVM_DEBUG(llvm::dbgs() << "Found last instance of op-type before consumer: ";
                     executeOp.getPayloadOp()->print(llvm::dbgs(), asmState);
                     llvm::dbgs() << "\n");
          break;
        };
      };



      if (!workItem.producerUnit)
        continue;

      StringAttr producerUnitName = workItem.producerUnit.getLeafReference();

      auto unitIt = map.find(producerUnitName);
      if (unitIt == map.end()) {
        LLVM_DEBUG(llvm::dbgs() << "Hardware unit schedule not found" << "\n");
        continue;
      }

      DenseMap<Operation *, int64_t> &unitSchedule = unitIt->second;

      if (!lastInstanceOfType) {
        LLVM_DEBUG(llvm::dbgs() << "LastUse not found" << "\n");
        continue;
      }

      auto lastUseIt = unitSchedule.find(lastInstanceOfType.getPayloadOp());
      if (lastUseIt == unitSchedule.end()) {
        LLVM_DEBUG(llvm::dbgs() << "LastUse not found in schedule: ";
                   lastInstanceOfType->print(llvm::dbgs(), asmState);
                   llvm::dbgs() << "\n");
        continue;
      }
      int64_t lastUseCount = lastUseIt->second;

      auto producerIt = unitSchedule.find(workItem.producerOp.getPayloadOp());
      if (producerIt == unitSchedule.end()) {
        LLVM_DEBUG(llvm::dbgs() << "ProducerCount not found in schedule: ";
                   workItem.producerOp.getPayloadOp()->print(llvm::dbgs(), asmState);
                   llvm::dbgs() << "\n");
        continue;
      }
      int64_t producerCount = producerIt->second;

      int64_t diff = lastUseCount - producerCount;

      LLVM_DEBUG(llvm::dbgs() << "DIFF : " << diff << "\n");


      auto symbol = workItem.producerOp.getPayloadOp()->getAttrOfType<SymbolRefAttr>("hw_unit");
      Operation *hwOp = symbol
        ? SymbolTable::lookupNearestSymbolFrom(workItem.producerOp.getPayloadOp(), symbol)
        : nullptr;

      auto queueDepth = findQueueDepth(hwOp);

      if(diff >= queueDepth) {
        LLVM_DEBUG(llvm::dbgs() << "SEM IS SWEEPABLE" << "\n");
        auto semAlloc = findSemAlloc(workItem.producerOp);

        auto checkResult = checkChain(semAlloc);

        if(checkResult.first.empty() && checkResult.second.empty()) {
          LLVM_DEBUG(llvm::dbgs() << "CULLING SEM" << "\n");
          semAlloc.cullSemaphore(); 
        }
      }

    };
  };


  static DenseMap<StringAttr, DenseMap<Operation *, int64_t>> buildSchedule(ModuleOp moduleOp) {

    SmallVector<eaac::HardwareScheduleOp> schedules;
    moduleOp.walk([&](HardwareScheduleOp s) { schedules.push_back(s); });
    assert(schedules.size() == 1); // Only a single schedule pr module.

    eaac::HardwareScheduleOp schedule = schedules.pop_back_val();

    StringRef name = SymbolTable::getSymbolName(schedule).getValue();

    if(!name.empty())
      LLVM_DEBUG(llvm::dbgs() << "Schedule: " << name << "\n");

    DenseMap<StringAttr, DenseMap<Operation *, int64_t>> map;

    schedule.getBody().walk([&](Operation *hwOp) {

      // Skip ops that aren't themselves symbol-defining hardware units.
      StringAttr hwName = SymbolTable::getSymbolName(hwOp);
      if (!hwName)
        return;

      // Build set of operations conforming to speific hardware op
      llvm::SetVector<mlir::Operation *> userOps;
      auto uses = SymbolTable::getSymbolUses(hwOp, moduleOp);
      for(auto user : *uses) {
        if(Operation *userOp = user.getUser())
          userOps.insert(userOp);
      }

      // Use actual walk to assign ordering.
      int64_t count = 0;
      moduleOp.walk([&](Operation *op) {
        if(userOps.contains(op)) {
          map[hwName][op] = count;
          count += 1;
        }
      });
    });

    return map;
  };
};



} // anonymous namespace

std::unique_ptr<Pass> createSemOptimizeQueuePass() {
  return std::make_unique<SemOptimizeQueuePass>();
}

} // namespace eaac
} // namespace mlir
