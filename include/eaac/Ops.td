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
// DmaStartOp - Initiate an asynchronous DMA transfer
//===----------------------------------------------------------------------===//

def DmaStartOp : EAAC_Op<"dma_start", [MemoryEffects<[MemRead<DefaultResource>,
                                                        MemWrite<DefaultResource>]>]> {
  let summary = "Start an asynchronous DMA transfer between memory tiers";
  let description = [{
    Initiates an asynchronous DMA transfer from `src` to `dst`. Returns an
    index token that can be passed to `eaac.dma_wait` to block until the
    transfer completes.

    Example:
    ```mlir
    %token = eaac.dma_start(%src, %dst) : memref<128x128xi8>, memref<128x128xi8> -> index
    eaac.dma_wait(%token)
    ```
  }];

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst);
  let results = (outs Index:$token);

  let assemblyFormat = [{
    `(` $src `,` $dst `)` attr-dict `:` type($src) `,` type($dst) `->` type($token)
  }];
}

//===----------------------------------------------------------------------===//
// DmaWaitOp - Wait for an asynchronous DMA transfer to complete
//===----------------------------------------------------------------------===//

def DmaWaitOp : EAAC_Op<"dma_wait"> {
  let summary = "Wait for an asynchronous DMA transfer to complete";
  let description = [{
    Blocks until the DMA transfer identified by `token` has completed.

    Example:
    ```mlir
    eaac.dma_wait(%token)
    ```
  }];

  let arguments = (ins Index:$token);

  let assemblyFormat = [{
    `(` $token `)` attr-dict
  }];
}

#endif // EAAC_OPS
