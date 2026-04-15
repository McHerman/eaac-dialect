//===- Ops.td - EAAC dialect operations --------------------*- tablegen -*-===//
//
// Defines the operations of the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_OPS
#define EAAC_OPS

include "eaac/EAAC.td"
include "mlir/Dialect/Async/IR/AsyncTypes.td"
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

def SemAllocOp : EAAC_Op<"sem_alloc"> {
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
  let results = (outs Index:$semaphore);

  let assemblyFormat = [{
    `(` $empty_count `,` $full_count `)` attr-dict `:` type($semaphore)
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

  let arguments = (ins Index:$semaphore);

  let assemblyFormat = [{
    `(` $semaphore `)` attr-dict `:` type($semaphore)
  }];
}

//===----------------------------------------------------------------------===//
// ComputeOp - Wrapper for compute operations with semaphore synchronization
//===----------------------------------------------------------------------===//

def ComputeOp : EAAC_Op<"compute", [NoTerminator]> {
  let summary = "Execute a compute region with semaphore synchronization";
  let description = [{
    Wraps compute operations (e.g. linalg ops) and associates them with
    semaphore pairs for producer-consumer synchronization. The hardware
    waits on input semaphores before execution and signals output
    semaphores on completion.

    Example:
    ```mlir
    eaac.compute(%sem0, %sem1, %sem2) {
      linalg.add ins(%a, %b : ...) outs(%c : ...)
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$semaphores);
  let regions = (region AnyRegion:$body);

  let assemblyFormat = [{
    `(` $semaphores `)` attr-dict-with-keyword regions
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
