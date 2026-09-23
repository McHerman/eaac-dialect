#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
//#include "llvm/ADT/STLExtras.h" // usually already pulled in transitively
#include <cassert>
#include <cstdint>
#include "llvm/ADT/TypeSwitch.h"


#define DEBUG_TYPE "eaac-sem-optimize"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_SEMOPTIMIZE
#include "eaac/Passes.h.inc"

namespace {





class SemOptimizePass
    : public impl::SemOptimizeBase<SemOptimizePass> {
public:
  using SemOptimizeBase::SemOptimizeBase;

  void runOnOperation() override {


    ModuleOp moduleOp = getOperation();

    processImplicitSerialization(moduleOp);

    //moduleOp.walk([&](func::FuncOp f) {
    //  processTransitivity(f);
    //});

  }

private:

  static void processImplicitSerialization(ModuleOp moduleOp) {

    SmallVector<eaac::HardwareScheduleOp> schedules;
    moduleOp.walk([&](HardwareScheduleOp s) { schedules.push_back(s); });
    assert(schedules.size() == 1); // Only a single schedule pr module.
    
    eaac::HardwareScheduleOp schedule = schedules.pop_back_val(); 

    StringRef name = SymbolTable::getSymbolName(schedule).getValue();

    if(!name.empty())
      LLVM_DEBUG(llvm::dbgs() << "Schedule: " << name << "\n");

    schedule.getBody().walk([&](Operation *hwOp) {

      llvm::DenseMap<mlir::Operation*, int64_t> work;

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
          work.try_emplace(op, count);
          count += 1;
        }
      });

      elliminateImplicit(work, hwOp);
    });
  };


  // Eliminate implicit
  // 1. Find require statements in execute block for given hardware op
  // 2. Find producer, and producer op
  // 3. Calculate distance 
  // 4. If greater than pipeline_depth of given hardware unit, remove semaphore
  //
  // Fuctions:
  // FindAcquire, takes op
  // FindRequire, takes op
  // CalcDistance, takes op and DenseMap
  // FindDepth, takes symbol
  // Remove sem, takes semaphore


  static void elliminateImplicit(const llvm::DenseMap<mlir::Operation*, int64_t> &work,
                                 mlir::Operation *hardwareOp) {
    for(auto &[key, value] : work) { // Iterate operations cheduled to same hardware unit
      auto parentExecute = dyn_cast<eaac::ExecuteOp>(key->getParentOp());
      if(!parentExecute)
        continue;
      llvm::SmallVector<eaac::SemRequireOp> requireOps = parentExecute.getRequireOps();
      // TODO check for aliasing chaining

      for(eaac::SemRequireOp requireOp : requireOps) {

        // Find link between producer and consumer semaphore
        eaac::SemAcquireOp acquireOp = requireOp.getProducer();
        auto pipeline_depth = findFunitDepth(hardwareOp);

        if(!acquireOp) { // Broadcast style semaphores
          LLVM_DEBUG(llvm::dbgs() << "Op has no producer, (Broadcast R-mode)" << "\n");

          bool dontCull = false;

          eaac::SemAllocOp parent = dyn_cast<eaac::SemAllocOp>(requireOp.getSemaphore().getDefiningOp());

          for (Value chainedSem : parent.getChainsFrom()) {
            LLVM_DEBUG(llvm::dbgs() << "Checking chain: sem_alloc=" << parent.getSemaphore()
                                     << " chained_sem=" << chainedSem << "\n");

            bool cull = checkChain(parent, chainedSem, work, pipeline_depth.value());

            if(!cull) {
              LLVM_DEBUG(llvm::dbgs() << "Broadcast semaphore cull cancelled: chains to other sem" << "\n");
              dontCull = true;
            }
          }

          for(auto use : parent.getResult().getUsers()) {
            eaac::SemAllocOp chained = dyn_cast<SemAllocOp>(use);

            if(chained) {
              if(!checkChain(chained, parent.getResult(), work, pipeline_depth.value())) {
                LLVM_DEBUG(llvm::dbgs() << "Broadcast semaphore cull cancelled: unable to clean chain from sem to other sem" << "\n");
                dontCull = true;
              }
            }
          }


          if(!dontCull) {
            LLVM_DEBUG(llvm::dbgs() << "CULLING BROADCAST SEM" << "\n");
            //cullSemaphore(parent);
            parent.cullSemaphore();
          }

        } else {

          auto producerOp = getPayloadOp(acquireOp);

          // Use schdule to find pipeline distance between ops
          auto distance = findDistance(producerOp, key, work);
          if(!distance.has_value())
            LLVM_DEBUG(llvm::dbgs() << "No distance val (cross unit async)" << "\n");

  
          if(pipeline_depth.has_value() && distance.has_value() ) {

            if(distance >= pipeline_depth) {
              LLVM_DEBUG(llvm::dbgs() << "Found implicit serialization" << "\n");
              // Still need to check if the semaphore takes any chaining inputs 
              // or provides chaining further down
              eaac::SemAllocOp parent = dyn_cast<eaac::SemAllocOp>(requireOp.getSemaphore().getDefiningOp());

              bool dontCull = false;
              
              for (Value chainedSem : parent.getChainsFrom()) {
                if(!checkChain(parent, chainedSem, work, pipeline_depth.value())) {
                  LLVM_DEBUG(llvm::dbgs() << "Semaphore cull cancelled: chains to other sem" << "\n");
                  dontCull = true;
                }
              }

              for(auto use : parent.getResult().getUsers()) {
                eaac::SemAllocOp chained = dyn_cast<SemAllocOp>(use);

                if(chained) {
                  if(!checkChain(chained, parent.getResult(), work, pipeline_depth.value())) {
                    LLVM_DEBUG(llvm::dbgs() << "Semaphore cull cancelled: unable to clean chain from sem to other sem" << "\n");
                    dontCull = true;
                  }
                }
              }

              if(!dontCull) {
                //cullSemaphore(parent);
                parent.cullSemaphore();
              }
            }else{
              LLVM_DEBUG(llvm::dbgs() << "No implicit serialization, distance:" << distance << "\n");
            }

          }
        }
      } 
    }
  };

  static std::optional<int64_t> findDistance(mlir::Operation *earlier,
                                              mlir::Operation *later,
                                              const llvm::DenseMap<mlir::Operation *, int64_t> &work) {
    auto laterIt = work.find(later);
    if(laterIt == work.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Later not found" << "\n");
      return std::nullopt;
    }

    auto earlierIt = work.find(earlier);
    if(earlierIt == work.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Earlier not found" << "\n");
      return std::nullopt;
    }
  
    int64_t laterTime = laterIt->second;
    int64_t earlierTime = earlierIt->second;
  
    return laterTime - earlierTime;
  }


  static std::optional<int64_t> findFunitDepth(mlir::Operation *op) {

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

    return funitIface.getPipelineDepth();
  }


  // The payload op of the eaac.execute region containing a semaphore op.
  static mlir::Operation *getPayloadOp(mlir::Operation *semOp) {
    if(!semOp)
      return nullptr;
    auto parent = dyn_cast<eaac::ExecuteOp>(semOp->getParentOp());
    return parent ? parent.getPayloadOp() : nullptr;
  }

  static bool checkChain(eaac::SemAllocOp op, Value chainedSem,
                          const llvm::DenseMap<mlir::Operation *, int64_t> &work,
                          int64_t depth) {

    llvm::SmallVector<eaac::SemRequireOp> chain;
    llvm::SmallVector<mlir::Operation *> chained;

    LLVM_DEBUG(llvm::dbgs() << "running chain check" << "\n");

    for (Operation *user : chainedSem.getUsers()) {
      llvm::TypeSwitch<Operation *>(user)
          .Case<eaac::SemRequireOp>([&](auto op) { chain.push_back(op); })
          .Default([&](Operation *op) {
            //op->emitError("unexpected user of chain semaphore");
          });
    }

    for (Operation *user : op.getResult().getUsers()) {
      llvm::TypeSwitch<Operation *>(user)
          .Case<eaac::SemRequireOp>([&](auto userOp) {
            if(op.getEventMode() == mlir::eaac::EventMode::R){ // Semaphore uses broadcast type chain
              chained.push_back(userOp); 
              //LLVM_DEBUG(llvm::dbgs() << "FOUND R-MODE CONSUMER" << "\n");
            }
          })
          .Case<eaac::SemAcquireOp>([&](auto userOp) {
            if(op.getEventMode() == mlir::eaac::EventMode::RW){ // Semaphore uses aliasing type chain 
              chained.push_back(userOp); 
              //LLVM_DEBUG(llvm::dbgs() << "FOUND RW-MODE CONSUMER" << "\n");
            }
          })
          .Default([&](Operation *op) {
            //op->emitError("unexpected user of chained semaphore");
          });
    }

    assert(chain.size() == 1);
    assert(chained.size() == 1);

    auto chainOp = getPayloadOp(chain.pop_back_val());
    auto chainedOp = getPayloadOp(chained.pop_back_val());

    auto distance = findDistance(chainOp, chainedOp, work);

    if(distance.has_value() && distance.value() >= depth){
      return true;
    };

    return false; 
  }


  //static void cullSemaphore(eaac::SemAllocOp op) {

  //  LLVM_DEBUG(llvm::dbgs() << "CULLING SEM!" << "\n");

  //  for(auto user : op.getResult().getUsers()) {
  //    if(!isa<eaac::SemAllocOp>(user)) {
  //      user->erase();
  //    } else {
  //      LLVM_DEBUG(llvm::dbgs() << "attempting to cull sem still chained to alloc, cancelled" << "\n");
  //    };
  //  }
  //  if(op.use_empty())
  //    op.erase();

  //  return; 
  //}

  /*
  static void processTransitivity(func::FuncOp funcOp){
    // Step 1: Number all operations
    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });

    llvm::DenseMap<eaac::SemRequireOp, eaac::SemAcquireOp> pairs;

    funcOp.getBody().walk([&](eaac::ExecuteOp executeOp) {  
      executeOp.getBody().walk([&](eaac::SemRequireOp requireOp) {

        eaac::SemAcquireOp producer = requireOp.getProducer();

        if(!producer)
          return WalkResult::skip(); 
         
        bool trans = transitivelyOrdered(requireOp, producer, opTime);

        if(trans) {
          //LLVM_DEBUG(llvm::dbgs() << "TRANS!! \n");
          auto [it, inserted] = pairs.try_emplace(requireOp, producer);
        };

        return WalkResult::advance(); 

      });
    });
  };

  static bool transitivelyOrdered(
      eaac::SemRequireOp later, eaac::SemAcquireOp earlier,
      const llvm::DenseMap<Operation *, int64_t> &opTime) {
    if (later == earlier)
      return true;
  
    auto earlierIt = opTime.find(earlier);
    if (earlierIt == opTime.end())
      return false;
    const int64_t earlierTime = earlierIt->second;
  
    llvm::SmallPtrSet<Operation *, 32> visited;
    //llvm::SmallVector<async::ExecuteOp, 16> work;
    llvm::SmallVector<eaac::SemRequireOp, 16> work;
    work.push_back(later);
  
    while (!work.empty()) {
      eaac::SemRequireOp cur = work.pop_back_val();
      for (Value sem : cur.getOperands()) {
        auto alloc = sem.getDefiningOp<eaac::SemAllocOp>();
        if (!alloc)
          continue;
  
        mlir::Operation *producerPtr = nullptr;
  
        for(auto user : alloc->getResults().getUsers()) {
          auto acquireOp = dyn_cast<eaac::SemAcquireOp>(user);
  
          if(acquireOp) {
            producerPtr = acquireOp.getOperation();
            continue;
          }
        }
  
        assert(producerPtr);
        auto producer = dyn_cast<eaac::SemAcquireOp>(producerPtr);
  
        if (producer == earlier)
          return true;
  
        auto it = opTime.find(producer);
        if (it == opTime.end() || it->second < earlierTime)
          continue; // can't reach `earlier` from here
        
        
        producer->getParentOp()->walk([&](eaac::SemRequireOp requireOp) {
          if (visited.insert(requireOp).second)
            work.push_back(requireOp);
        });
  
      }
    }
    return false;
  }
  */




};

} // anonymous namespace

std::unique_ptr<Pass> createSemOptimizePass() {
  return std::make_unique<SemOptimizePass>();
}

} // namespace eaac
} // namespace mlir
