#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
//#include "llvm/ADT/STLExtras.h" // usually already pulled in transitively
#include <cassert>
#include <cstdint>

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

    schedule.getBody().walk([&](Operation *op) {
      
      llvm::DenseMap<mlir::Operation*, int64_t> work;
      int64_t count = 0;

      auto uses = SymbolTable::getSymbolUses(op, moduleOp);
      for(auto user : *uses) {

        Operation *userOp = user.getUser();

        if(userOp){
          LLVM_DEBUG(llvm::dbgs() << "Found hardware user" << "\n");
          work.try_emplace(userOp, count);
          count += 1;
        }
      }
      elliminateImplicit(work, op);
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
      llvm::SmallVector<eaac::SemRequireOp> requireOps = findRequire(key);
      // TODO check for aliasing chaining
      
      for(eaac::SemRequireOp requireOp : requireOps) {

        // Find link between producer and consumer semaphore
        eaac::SemAcquireOp acquireOp = findProducer(requireOp);
        auto producerOp = findOp(acquireOp);

        // Use schdule to find pipeline distance between ops
        auto pipeline_depth = findFunitDepth(hardwareOp);
        auto distance = findDistance(key, producerOp, work);
        if(!distance.has_value())
          LLVM_DEBUG(llvm::dbgs() << "No distance val" << "\n");

  
        if(pipeline_depth.has_value() && distance.has_value() ) {

          LLVM_DEBUG(llvm::dbgs() << "Valid pipeline and distance vals" << "\n");

          if(distance >= pipeline_depth) {
            LLVM_DEBUG(llvm::dbgs() << "Found implicit serialization" << "\n");
            // Still need to check if the semaphore takes any chaining inputs 
            // or provides chaining further down
            //
            eaac::SemAllocOp parent = dyn_cast<eaac::SemAllocOp>(requireOp.getSemaphore().getDefiningOp());
            cullSemaphore(parent);
          }else{
            LLVM_DEBUG(llvm::dbgs() << "No implicit serialization, distance:" << distance << "\n");
          }

        }
      } 

    }
  };


  static mlir::Operation* findOp(mlir::Operation *op){
    eaac::ExecuteOp parentOp = dyn_cast<eaac::ExecuteOp>(op->getParentOp());


    if(!parentOp) {
      LLVM_DEBUG(llvm::dbgs() << "Incorrect parent op" << "\n");
      return nullptr;
    }

    llvm::SmallVector<Operation *> ops;

    for (Operation &bodyOp : parentOp.getBody().getOps()) {
      if(!isa<eaac::SemRequireOp>(bodyOp) && !isa<eaac::SemAcquireOp>(bodyOp)) {
        //LLVM_DEBUG(llvm::dbgs() << "Pushing op: " << bodyOp.getName() << "\n");
        ops.push_back(&bodyOp);
      }
    }

    if (ops.size() != 1) {
      return nullptr;
    }

    return ops.pop_back_val();
  };





  static llvm::SmallVector<eaac::SemRequireOp> findRequire(mlir::Operation *op){
    llvm::SmallVector<eaac::SemRequireOp> returnOps;

    eaac::ExecuteOp parentOp = dyn_cast<eaac::ExecuteOp>(op->getParentOp());

    if(parentOp) {
      parentOp.getBody().walk([&](eaac::SemRequireOp requireOp) {  
        returnOps.push_back(requireOp);
      });
    }

    return returnOps;
  };


  static eaac::SemAcquireOp findAcquire(mlir::Operation *op){

    eaac::SemAcquireOp returnOp;
    eaac::ExecuteOp parentOp = dyn_cast<eaac::ExecuteOp>(op->getParentOp());

    if(parentOp) {
      auto ops = parentOp.getBody().getOps<eaac::SemAcquireOp>();
      assert(llvm::range_size(ops) == 1); // Ensure a single acquire pr eaac execute region
      returnOp = *ops.begin();
    }
      
    return returnOp;
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
  
    if (laterTime > earlierTime)
      return laterTime - earlierTime;
    return std::nullopt;
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


  static void cullSemaphore(eaac::SemAllocOp op) {

    // Check whether semaphore chains from other sem
    for (Value chainedSem : op.getChainsFrom()) {
      LLVM_DEBUG(llvm::dbgs() << "Semaphore cull cancelled: chains from other sem" << "\n");
      return;
    }

    // Check if other sem chains current sem
    for (Operation *userOp : op.getSemaphore().getUsers()) {
      eaac::SemAllocOp allocOp = dyn_cast<eaac::SemAllocOp>(userOp);

      if(!allocOp)
        continue;

      // Slightly unnessecary check, but probably good futureproofing
      //if(allocOp.getChainsFrom() == op.getSemaphore()) {
      if (llvm::is_contained(allocOp.getChainsFrom(), op.getSemaphore())) {
        LLVM_DEBUG(llvm::dbgs() << "Semaphore cull cancelled: is chained by other sem" << "\n");
        return; 
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Semaphore culled" << "\n");

    for(auto user : op.getResult().getUsers()) {
      user->erase();
    }

    if(op.use_empty())
      op.erase();

    return; 
  }


  static void processTransitivity(func::FuncOp funcOp){
    // Step 1: Number all operations
    llvm::DenseMap<Operation *, int64_t> opTime;
    int64_t time = 0;
    funcOp.walk([&](Operation *op) { opTime[op] = time++; });

    llvm::DenseMap<eaac::SemRequireOp, eaac::SemAcquireOp> pairs;

    funcOp.getBody().walk([&](eaac::ExecuteOp executeOp) {  
      executeOp.getBody().walk([&](eaac::SemRequireOp requireOp) {

        eaac::SemAcquireOp producer = findProducer(requireOp);

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


  static eaac::SemAcquireOp findProducer(eaac::SemRequireOp op){
    eaac::SemAcquireOp returnOp;
    for(auto arg : op->getOperands()) {

      auto sem = arg.getDefiningOp()->getResults();

      for(auto user : sem.getUsers()){
        eaac::SemAcquireOp acquireOp = dyn_cast<eaac::SemAcquireOp>(user);

        if(!acquireOp)
          continue;

        //LLVM_DEBUG(llvm::dbgs() << "Found semaphore acquire \n");

        returnOp = acquireOp;
      }; 
    };

    return returnOp;
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




};

} // anonymous namespace

std::unique_ptr<Pass> createSemOptimizePass() {
  return std::make_unique<SemOptimizePass>();
}

} // namespace eaac
} // namespace mlir
