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
  let parameters = (ins "int64_t":$addr, "int64_t":$gen);
  let assemblyFormat = "`<` `addr` `=` $addr `,` `gen` `=` $gen `>`";
  let description = [{
    `!eaac.semaphore` represents a handle a hardware semaphore
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

class EAAC_Funit_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<EAAC_Dialect, name, traits> {
  let mnemonic = typeMnemonic;

  let parameters = (ins "unsigned":$pipeline_depth);
  let assemblyFormat = "`<` $pipeline_depth `>`";

  let genMnemonicAlias = 1;
}

def Load_Type : EAAC_Funit_Type<"Load", "load"> {
  let summary = "Load unit";
  let description = [{
    EAAC load unit
  }];
}

def Store_Type : EAAC_Funit_Type<"Store", "store"> {
  let summary = "Store unit";
  let description = [{
    EAAC load unit
  }];
}


def RISC_Type : EAAC_Funit_Type<"RISC", "risc"> {
  let summary = "Store unit";
  let description = [{
    EAAC RISC-V unit
  }];
}

def GEMM_Type : EAAC_Funit_Type<"GEMM", "gemm"> {
  let summary = "GEMM unit computing matmuls";
  let description = [{
    Capable of computing linalg.matmul operations
  }];
}

#endif // EAAC_TYPES
