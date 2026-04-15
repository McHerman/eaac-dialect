//===- Types.td - EAAC dialect types -----------------------*- tablegen -*-===//
//
// Defines the types of the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TYPES
#define EAAC_TYPES

include "mlir/IR/AttrTypeBase.td"
include "eaac/EAAC.td"

def EAAC_SemaphoreType : TypeDef<EAAC_Dialect, "Semaphore"> {
  let mnemonic = "semaphore";
  let summary = "Handle to a hardware semaphore pair";
  let description = [{
    `!eaac.semaphore` represents a handle to one of the 16 hardware semaphore
    pairs used for producer-consumer synchronization between DMA and compute
    operations.
  }];
}

#endif // EAAC_TYPES
