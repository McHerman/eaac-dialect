//===- ScheduleInterfacesImpls.cpp - ScheduleInterface external models --===//
//
// Attaches ScheduleInterface to ops via external models, without modifying
// the ops' own ODS definitions.
//
//===----------------------------------------------------------------------===//

#include "eaac/Dialect.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "eaac-schedule-interfaces"

using namespace mlir;
using namespace llvm;

namespace {

/// External model implementing ScheduleInterface for eaac::MatmulOp.
/// Template args: <ThisModelClass, ConcreteOpType>
struct MatmulScheduleModel
    : public eaac::detail::ScheduleInterfaceInterfaceTraits::ExternalModel<
          MatmulScheduleModel, linalg::MatmulOp> {
  Operation *checkSchedule(Operation *op, Region *region) const {

    llvm::SmallVector<eaac::HardwareAllocOp> hardware_units;

    region->front().walk([&](eaac::HardwareAllocOp funit) {
      if (isa<eaac::GEMMType>(funit.getFunit()))
        hardware_units.push_back(funit);
    });

    // TODO, impement more intelligment scheduling for multible hardware units

    if(!hardware_units.empty()) {
      return hardware_units.pop_back_val();
    } else {
      return nullptr;
    }
  }
};

struct AddScheduleModel
    : public eaac::detail::ScheduleInterfaceInterfaceTraits::ExternalModel<
          AddScheduleModel, linalg::AddOp> {
  Operation *checkSchedule(Operation *op, Region *region) const {

    llvm::SmallVector<eaac::HardwareAllocOp> hardware_units;

    region->front().walk([&](eaac::HardwareAllocOp funit) {
      if (isa<eaac::RISCType>(funit.getFunit()))
        hardware_units.push_back(funit);
    });

    // TODO, impement more intelligment scheduling for multible hardware units

    if(!hardware_units.empty()) {
      return hardware_units.pop_back_val();
    } else {
      return nullptr;
    }
  }
};

struct GenericScheduleModel
    : public eaac::detail::ScheduleInterfaceInterfaceTraits::ExternalModel<
          GenericScheduleModel, linalg::GenericOp> {
  Operation *checkSchedule(Operation *op, Region *region) const {

    llvm::SmallVector<eaac::HardwareAllocOp> hardware_units;

    region->front().walk([&](eaac::HardwareAllocOp funit) {
      if (isa<eaac::RISCType>(funit.getFunit()))
        hardware_units.push_back(funit);
    });

    // TODO, impement more intelligment scheduling for multible hardware units

    if(!hardware_units.empty()) {
      return hardware_units.pop_back_val();
    } else {
      return nullptr;
    }
  }
};


} // namespace

void mlir::eaac::registerScheduleOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, eaac::EAACDialect *eaacDialect,
          linalg::LinalgDialect *linalgDialect) {
        linalg::MatmulOp::attachInterface<MatmulScheduleModel>(*ctx);
        linalg::AddOp::attachInterface<AddScheduleModel>(*ctx);
        linalg::GenericOp::attachInterface<GenericScheduleModel>(*ctx);
      });
}
