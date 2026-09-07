#include "eaac/Dialect.h"
#include "eaac/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "eaac-strip-schedule"

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_STRIPSCHEDULE
#include "eaac/Passes.h.inc"

namespace {

class StripSchedulePass
    : public impl::StripScheduleBase<StripSchedulePass> {
public:
  using StripScheduleBase::StripScheduleBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

      
    moduleOp.getBody()->walk([&](eaac::HardwareScheduleOp op) {
      op.erase();
    });

  }
};

} // anonymous namespace

std::unique_ptr<Pass> createStripSchedulePass() {
  return std::make_unique<StripSchedulePass>();
}

} // namespace eaac
} // namespace mlir
