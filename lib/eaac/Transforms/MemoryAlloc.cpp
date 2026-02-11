//===- MemoryAlloc.cpp - Assign static offsets to memory elements -===//
//
// Pass to collect memref allocations and print their liveness intervals
// using MLIR's built-in Liveness analysis.
//
//===----------------------------------------------------------------------===//

#include "eaac/Passes.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>

#include "minimalloc.h"
#include "solver.h"           

namespace mlir {
namespace eaac {

#define GEN_PASS_DEF_MEMORYALLOC
#include "eaac/Passes.h.inc"

namespace {

/// Computes the size in bytes for a memref type.
uint64_t getMemRefSizeInBytes(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;

  int64_t numElements = 1;
  for (int64_t dim : type.getShape())
    numElements *= dim;

  Type elementType = type.getElementType();
  uint64_t elementBits = elementType.isIntOrFloat()
      ? elementType.getIntOrFloatBitWidth()
      : 32;

  return numElements * ((elementBits + 7) / 8);
}

class MemoryAllocPass
    : public impl::MemoryAllocBase<MemoryAllocPass> {
public:
  using MemoryAllocBase::MemoryAllocBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Create minimalloc problem
    minimalloc::Problem problem;

    problem.capacity = 1000000000;
    int bufferIdx = 0;

    module.walk([&](func::FuncOp funcOp) {

      // Step 1: Number all operations
      llvm::DenseMap<Operation *, int64_t> opTime;
      int64_t time = 0;
      funcOp.walk([&](Operation *op) { opTime[op] = time++; });

      // Step 2: Compute liveness using MLIR's analysis
      Liveness liveness(funcOp);

      // Step 3: Process each allocation
      funcOp.walk([&](memref::AllocOp allocOp) {
        Value memref = allocOp.getResult();
        Operation *startOp = allocOp;
        Operation *endOp = allocOp;

        // Find the last use across all blocks using Liveness
        for (Block &block : funcOp.getBody()) {
          const LivenessBlockInfo *blockInfo = liveness.getLiveness(&block);
          if (!blockInfo)
            continue;

          if (Operation *end = blockInfo->getEndOperation(memref, startOp)) {
            int64_t endTime = opTime[end];
            int64_t currentEndTime = opTime[endOp];
            if (endTime > currentEndTime)
              endOp = end;
          }
        }

        // Look up times for start and end operations
        int64_t startTime = opTime[startOp];
        int64_t endTime = opTime[endOp];

        MemRefType type = allocOp.getType();
        uint64_t size = getMemRefSizeInBytes(type);

        int64_t sizeInBytes = static_cast<int64_t>(size);

        problem.buffers.push_back(minimalloc::Buffer{
            .id = "buf" + std::to_string(bufferIdx++),
            .lifespan = {startTime, endTime + 1},  // half-open interval
            .size = sizeInBytes,
            .alignment = 1,
            .gaps = {},
            .offset = std::nullopt,
            .hint = std::nullopt
        });

      });
    });

    // Solve
    minimalloc::Solver solver;
    auto result = solver.Solve(problem);

    llvm::errs() << "========================================\n";
    llvm::errs() << "Static Memory Allocation Results\n";
    llvm::errs() << "========================================\n\n";

    if (result.ok()) {
      minimalloc::Solution solution = *result;

      llvm::errs() << "Buffers: " << problem.buffers.size() << "\n";
      llvm::errs() << "Total memory required: " << solution.height << " bytes\n\n";

      for (size_t i = 0; i < problem.buffers.size(); ++i) {
        const auto &buf = problem.buffers[i];
        minimalloc::Offset offset = solution.offsets[i];

        llvm::errs() << "Buffer " << i << ":\n";
        llvm::errs() << "  ID:       " << buf.id << "\n";
        llvm::errs() << "  Size:     " << buf.size << " bytes\n";
        llvm::errs() << "  Lifespan: [" << buf.lifespan.lower() << ", "
                     << buf.lifespan.upper() << ")\n";
        llvm::errs() << "  Offset:   " << offset << "\n";
        llvm::errs() << "  Range:    [" << offset << ", "
                     << (offset + buf.size) << ")\n\n";
      }
    } else {
      llvm::errs() << "Solver failed: allocation not possible\n";
    }

    llvm::errs() << "========================================\n";
  }
};

} // anonymous namespace

std::unique_ptr<Pass> createMemoryAllocPass() {
  return std::make_unique<MemoryAllocPass>();
}

} // namespace eaac
} // namespace mlir
