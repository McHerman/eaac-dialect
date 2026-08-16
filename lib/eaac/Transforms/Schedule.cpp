#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "eaac-schedule"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_SCHEDULE
#include "eaac/Passes.h.inc"

namespace {

class SchedulePass
    : public impl::ScheduleBase<SchedulePass> {
public:
  using ScheduleBase::ScheduleBase;

  void runOnOperation() override {

    ModuleOp moduleOp = getOperation();

    SmallVector<func::FuncOp> functions;
    moduleOp.walk([&](func::FuncOp f) { functions.push_back(f);});

    SmallVector<eaac::HardwareScheduleOp> schedules;
    moduleOp.walk([&](eaac::HardwareScheduleOp s) { schedules.push_back(s); });
    assert(schedules.size() == 1); // Only a single schedule pr module.

    eaac::HardwareScheduleOp schedule = schedules.pop_back_val();

    LLVM_DEBUG(llvm::dbgs() << "running scheduling \n");

    for(auto funcOp : functions){
      funcOp.getBody().walk([&](Operation *op) {
        if (auto iface = dyn_cast<ScheduleInterface>(op)) {

          Operation *funit = iface.checkSchedule(&schedule.getBody());

          // Units live inside the named eaac.schedule symbol table, so
          // consumers must use a qualified reference: @schedule::@unit.
          op->setAttr("hw_unit",
              SymbolRefAttr::get(SymbolTable::getSymbolName(schedule),
                  {FlatSymbolRefAttr::get(SymbolTable::getSymbolName(funit))}));
        }
      });
    }
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createSchedulePass() {
  return std::make_unique<SchedulePass>();
}

} // namespace eaac
} // namespace mlir
