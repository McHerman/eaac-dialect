//===- Ops.td - EAAC dialect operations --------------------*- tablegen -*-===//
//
// Defines the operations of the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_OPS
#define EAAC_OPS

include "eaac/EAAC.td"
include "eaac/Types.td"
include "mlir/Dialect/Async/IR/AsyncTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// DmaStartOp - Initiate an asynchronous DMA transfer
//===----------------------------------------------------------------------===//

def DmaStartOp : EAAC_Op<"dma_start", [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Start an asynchronous DMA transfer between memory tiers";
  let description = [{
    Initiates an asynchronous DMA transfer from `src` to `dst`.
    Optional semaphore arguments provide producer-consumer synchronization.
    The hardware signals the semaphore(s) on transfer completion.

    Example:
    ```mlir
    eaac.dma_start(%src, %dst) : memref<128x128xi8>, memref<128x128xi8>
    eaac.dma_start(%src, %dst, %sem) : memref<128x128xi8>, memref<128x128xi8>, index
    ```
  }];

  let arguments = (ins AnyMemRef:$src, AnyMemRef:$dst, Variadic<Index>:$semaphores);

  let assemblyFormat = [{
    `(` $src `,` $dst (`,` $semaphores^)? `)` attr-dict
    `:` type($src) `,` type($dst) (`,` type($semaphores)^)?
  }];
}

//===----------------------------------------------------------------------===//
// SemAllocOp - Allocate and initialize a hardware semaphore pair
//===----------------------------------------------------------------------===//

def SemAllocOp : EAAC_Op<"sem_alloc", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Allocate and initialize a hardware semaphore pair";
  let description = [{
    Allocates a semaphore pair from the hardware bank and initializes it
    with the given empty and full counts. Returns an index identifying
    the semaphore pair.

    For unbounded buffers: empty_count = large value, full_count = 0.
    For bounded buffers:   empty_count = N,           full_count = 0.

    Example:
    ```mlir
    %sem = eaac.sem_alloc(%empty, %full) : index
    ```
  }];

  let arguments = (ins Index:$empty_count, Index:$full_count);
  let results = (outs EAAC_SemaphoreType:$semaphore);

  let assemblyFormat = [{
    `(` $empty_count `,` $full_count `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SemDeallocOp - Release a hardware semaphore pair
//===----------------------------------------------------------------------===//

def SemDeallocOp : EAAC_Op<"sem_dealloc"> {
  let summary = "Release a hardware semaphore pair back to the bank";
  let description = [{
    Returns the semaphore slot to the free pool. Must be called after
    all producers and consumers using this semaphore are complete.

    Example:
    ```mlir
    eaac.sem_dealloc(%sem) : index
    ```
  }];

  let arguments = (ins EAAC_SemaphoreType:$semaphore);

  let assemblyFormat = [{
    `(` $semaphore `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// ExecuteOp - Region wrapper for dependency-guarded operations
//===----------------------------------------------------------------------===//

def ExecuteOp : EAAC_Op<"execute", [NoTerminator]> {
  let summary = "Execute a region with dependency-guarded operations";
  let description = [{
    Wraps operations that participate in producer-consumer synchronization.
    Dependencies are expressed by eaac.require or eaac.sem_require ops
    inside the body rather than as arguments to the region.

    Example:
    ```mlir
    eaac.execute {
      eaac.sem_require %sem0, %buf_a : memref<8x8xf32>
      eaac.dma_start(%src, %buf_a) : memref<8x8xf32>, memref<8x8xf32>
    }
    ```
  }];

  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    attr-dict-with-keyword regions
  }];
}

//===----------------------------------------------------------------------===//
// SemRequireOp - Explicit semaphore-to-memref dependency pairing
//===----------------------------------------------------------------------===//

def SemRequireOp : EAAC_Op<"sem_require", [
    TypesMatchWith<"result type matches memref type",
                   "memref", "result", "$_self">]> {
  let summary = "Pair a hardware semaphore with a specific memref";
  let description = [{
    Declares that `memref` is guarded by `semaphore` for producer-consumer
    synchronization. Returns the same memref as a new SSA value.

    Example:
    ```mlir
    %ready = eaac.sem_require %sem, %buf : memref<128x128xi8>
    ```
  }];

  let arguments = (ins EAAC_SemaphoreType:$semaphore, AnyMemRef:$memref);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $semaphore `,` $memref attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// SemAcquireOp - Mark a producer's semaphore-to-memref pairing
//===----------------------------------------------------------------------===//

def SemAcquireOp : EAAC_Op<"sem_acquire", [
    TypesMatchWith<"result type matches memref type",
                   "memref", "result", "$_self">]> {
  let summary = "Mark that a producer signals this semaphore on completion";
  let description = [{
    Declares that the producer writing to `memref` will signal `semaphore`
    upon completion. Returns the same memref as a new SSA value.

    This is the producer-side counterpart to `eaac.sem_require`, which is
    the consumer-side wait.

    Example:
    ```mlir
    %acquired = eaac.sem_acquire %sem, %buf : memref<128x128xi8>
    eaac.dma_start(%src, %acquired) : memref<128x128xi8>, memref<128x128xi8>
    ```
  }];

  let arguments = (ins EAAC_SemaphoreType:$semaphore, AnyMemRef:$memref);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $semaphore `,` $memref attr-dict `:` type($memref)
  }];
}

//===----------------------------------------------------------------------===//
// RequireOp - Explicit token-to-memref dependency pairing
//===----------------------------------------------------------------------===//

def RequireOp : EAAC_Op<"require", [
    TypesMatchWith<"result type matches memref type",
                   "memref", "result", "$_self">]> {
  let summary = "Pair an async token dependency with a specific memref";
  let description = [{
    Declares that `memref` is only safe to use after `token` has resolved.
    Returns the same memref as a new SSA value, threading the dependency
    through the use-def chain.

    Example:
    ```mlir
    %ready = eaac.require %token, %buf : memref<128x128xi8>
    linalg.generic ins(%ready : memref<128x128xi8>) ...
    ```
  }];

  let arguments = (ins Async_TokenType:$token, AnyMemRef:$memref);
  let results = (outs AnyMemRef:$result);

  let assemblyFormat = [{
    $token `,` $memref attr-dict `:` type($memref)
  }];
}

#endif // EAAC_OPS
