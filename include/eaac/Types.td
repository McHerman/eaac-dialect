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

// Memory space attribute carried by memref types for eaac-allocated buffers.
// `tier` identifies the memory hierarchy level; `offset` is the hardware
// base address assigned by the memory allocator.
def EAAC_MemSpaceAttr : AttrDef<EAAC_Dialect, "MemSpace"> {
  let mnemonic = "mem";
  let summary = "EAAC hardware memory space (tier + base offset)";
  let parameters = (ins "int64_t":$tier, "int64_t":$offset);
  let assemblyFormat = "`<` `tier` `=` $tier `,` `offset` `=` $offset `>`";
}

#endif // EAAC_TYPES
