//===- Ops.td - EAAC dialect operations --------------------*- tablegen -*-===//
//
// Defines the operations of the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_OPS
#define EAAC_OPS

include "eaac/EAAC.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// ExeOp - Maps to int_eaac_exe intrinsic
//===----------------------------------------------------------------------===//

def EAAC_ExeOp : EAAC_Op<"exe"> {
  let summary = "Execute operation on accelerator";
  let description = [{
    Executes a computation on the EAAC accelerator.

    Maps to the LLVM intrinsic:
      int_eaac_exe(addrd, addrs1, addrs2, size, mode, op2, op, func)

    Arguments:
      - addrd:  Destination address (16-bit)
      - addrs1: Source address 1 (16-bit)
      - addrs2: Source address 2 (16-bit)
      - size:   Operation size (8-bit)
      - mode:   Mode flag (1-bit)
      - op2:    Secondary operation flag (1-bit)
      - op:     Operation type (2-bit, stored as 8-bit)
      - func:   Function selector (4-bit, stored as 8-bit)

    Example:
      eaac.exe %addrd, %addrs1, %addrs2, %size, %mode, %op2, %op, %func
  }];

  let arguments = (ins
    I16:$addrd,
    I16:$addrs1,
    I16:$addrs2,
    I8:$size,
    I1:$mode,
    I1:$op2,
    I8:$op,
    I8:$func
  );

  let results = (outs);

  let assemblyFormat = [{
    $addrd `,` $addrs1 `,` $addrs2 `,` $size `,` $mode `,` $op2 `,` $op `,` $func attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// LoadOp - Maps to int_eaac_load intrinsic
//===----------------------------------------------------------------------===//

def EAAC_LoadOp : EAAC_Op<"load"> {
  let summary = "Load data to accelerator";
  let description = [{
    Loads data from memory to the EAAC accelerator.

    Maps to the LLVM intrinsic:
      int_eaac_load(addr, size, mode, op, func)

    Arguments:
      - addr: Memory address (32-bit)
      - size: Load size (8-bit)
      - mode: Mode flag (1-bit)
      - op:   Operation flag (1-bit)
      - func: Function selector (4-bit, stored as 8-bit)

    Example:
      eaac.load %addr, %size, %mode, %op, %func
  }];

  let arguments = (ins
    I32:$addr,
    I8:$size,
    I1:$mode,
    I1:$op,
    I8:$func
  );

  let results = (outs);

  let assemblyFormat = [{
    $addr `,` $size `,` $mode `,` $op `,` $func attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// StoreOp - Maps to int_eaac_store intrinsic
//===----------------------------------------------------------------------===//

def EAAC_StoreOp : EAAC_Op<"store"> {
  let summary = "Store data from accelerator";
  let description = [{
    Stores data from the EAAC accelerator to memory.

    Maps to the LLVM intrinsic:
      int_eaac_store(addr, size, op, func)

    Arguments:
      - addr: Memory address (32-bit)
      - size: Store size (8-bit)
      - op:   Operation flag (1-bit)
      - func: Function selector (4-bit, stored as 8-bit)

    Example:
      eaac.store %addr, %size, %op, %func
  }];

  let arguments = (ins
    I32:$addr,
    I8:$size,
    I1:$op,
    I8:$func
  );

  let results = (outs);

  let assemblyFormat = [{
    $addr `,` $size `,` $op `,` $func attr-dict
  }];
}

#endif // EAAC_OPS
