//===- EAACToFlatbuffer.cpp - EAAC to FlatBuffer translation --------------===//
//
// Translates a fully-lowered EAAC MLIR module into a FlatBuffer binary.
//
//===----------------------------------------------------------------------===//

#include "eaac/Target/EAACToFlatbuffer.h"
#include "eaac/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "flatbuffers/flatbuffers.h"
#include "eaac_program_generated.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::eaac;
namespace fb = eaac_fb;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Convert an MLIR integer/float type to the FlatBuffer ElementType enum.
static fb::ElementType convertElementType(Type type) {
  if (type.isInteger(8))
    return fb::ElementType_I8;
  if (type.isInteger(16))
    return fb::ElementType_I16;
  if (type.isInteger(32))
    return fb::ElementType_I32;
  if (type.isInteger(64))
    return fb::ElementType_I64;
  if (type.isF16())
    return fb::ElementType_F16;
  if (type.isF32())
    return fb::ElementType_F32;
  if (type.isF64())
    return fb::ElementType_F64;
  return fb::ElementType_I8; // fallback
}

/// Walk through SemAcquire/SemRequire wrappers to the underlying buffer-defining
/// SSA value (a memref.alloc result or memref.get_global result). The returned
/// Value is the canonical identity used to dedupe buffer references.
static Value resolveUnderlyingMemref(Value memref) {
  Value current = memref;
  while (current) {
    if (auto reqOp = current.getDefiningOp<SemRequireOp>()) {
      current = reqOp.getMemref();
      continue;
    }
    if (auto acqOp = current.getDefiningOp<SemAcquireOp>()) {
      current = acqOp.getMemref();
      continue;
    }
    break;
  }
  return current;
}

/// Build a FlatBuffer BufferRef from an already-resolved underlying memref.
/// Reads eaac.offset / eaac.tier from the defining op and constant_name from
/// memref.get_global if present.
static flatbuffers::Offset<fb::BufferRef>
buildBufferRef(flatbuffers::FlatBufferBuilder &builder, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());

  uint64_t offset = 0;
  uint8_t tier = 0;
  std::string constantName;

  if (Operation *defOp = memref.getDefiningOp()) {
    if (auto attr = defOp->getAttrOfType<IntegerAttr>("eaac.offset"))
      offset = attr.getInt();
    if (auto attr = defOp->getAttrOfType<IntegerAttr>("eaac.tier"))
      tier = attr.getInt();
    if (auto getGlobalOp = dyn_cast<memref::GetGlobalOp>(defOp))
      constantName = getGlobalOp.getName().str();
  }

  std::vector<uint32_t> shape;
  for (int64_t dim : memrefType.getShape())
    shape.push_back(static_cast<uint32_t>(dim));

  auto elemType = convertElementType(memrefType.getElementType());
  const char *constNamePtr = constantName.empty() ? nullptr : constantName.c_str();
  return fb::CreateBufferRefDirect(builder, offset, tier, &shape, elemType,
                                   constNamePtr);
}

/// Per-function table mapping each distinct underlying memref SSA value to a
/// stable uint32 buffer_id. The id is the index the buffer will occupy in
/// Function.buffers when the table is materialized.
namespace {
struct BufferTable {
  llvm::DenseMap<Value, uint32_t> idMap;
  std::vector<Value> order;

  uint32_t getOrAssign(Value memref) {
    Value resolved = resolveUnderlyingMemref(memref);
    auto it = idMap.find(resolved);
    if (it != idMap.end())
      return it->second;
    uint32_t id = static_cast<uint32_t>(order.size());
    idMap[resolved] = id;
    order.push_back(resolved);
    return id;
  }
};
} // namespace

/// Get the hardware semaphore address from a sem_alloc op via eaac.sem_addr.
static uint16_t getSemAddress(Value semaphore) {
  auto allocOp = semaphore.getDefiningOp<SemAllocOp>();
  if (!allocOp)
    return 0;
  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("eaac.sem_addr"))
    return attr.getInt();
  return 0;
}

/// Get the per-address generation tag of a sem_alloc via eaac.sem_gen.
static uint16_t getSemGeneration(Value semaphore) {
  auto allocOp = semaphore.getDefiningOp<SemAllocOp>();
  if (!allocOp)
    return 0;
  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("eaac.sem_gen"))
    return attr.getInt();
  return 0;
}

/// Resolve a list of chains_from operands into their hardware addresses.
static std::vector<uint16_t> getChainAddresses(ValueRange chains) {
  std::vector<uint16_t> addrs;
  addrs.reserve(chains.size());
  for (Value c : chains)
    addrs.push_back(getSemAddress(c));
  return addrs;
}

/// Try to extract a constant index value from an SSA value.
static uint32_t getConstantIndex(Value val) {
  if (auto constOp = val.getDefiningOp<arith::ConstantIndexOp>())
    return constOp.value();
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Main translation
//===----------------------------------------------------------------------===//

/// Serialize a single eaac.execute block into an Execute FlatBuffer table.
static flatbuffers::Offset<fb::Execute>
serializeExecute(flatbuffers::FlatBufferBuilder &builder, ExecuteOp execOp,
                 BufferTable &buffers) {
  std::vector<flatbuffers::Offset<fb::SemDep>> acquires;
  std::vector<flatbuffers::Offset<fb::SemDep>> semRequires;
  flatbuffers::Offset<void> payloadOffset;
  fb::ExecutePayload payloadType = fb::ExecutePayload_NONE;

  for (Operation &op : execOp.getBody().front()) {
    if (auto acqOp = dyn_cast<SemAcquireOp>(op)) {
      uint32_t bufId = buffers.getOrAssign(acqOp.getMemref());
      acquires.push_back(fb::CreateSemDepDirect(
          builder, getSemAddress(acqOp.getSemaphore()), bufId,
          /*chain_addresses=*/nullptr,
          getSemGeneration(acqOp.getSemaphore())));
    } else if (auto reqOp = dyn_cast<SemRequireOp>(op)) {
      uint32_t bufId = buffers.getOrAssign(reqOp.getMemref());
      auto chainAddrs = getChainAddresses(reqOp.getChainsFrom());
      semRequires.push_back(fb::CreateSemDepDirect(
          builder, getSemAddress(reqOp.getSemaphore()), bufId, &chainAddrs,
          getSemGeneration(reqOp.getSemaphore())));
    } else if (auto dmaOp = dyn_cast<DmaStartOp>(op)) {
      uint32_t src = buffers.getOrAssign(dmaOp.getSrc());
      uint32_t dst = buffers.getOrAssign(dmaOp.getDst());
      payloadType = fb::ExecutePayload_DmaStart;
      payloadOffset = fb::CreateDmaStart(builder, src, dst).Union();
    } else if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      uint32_t src0 = buffers.getOrAssign(matmulOp.getSrc0());
      uint32_t src1 = buffers.getOrAssign(matmulOp.getSrc1());
      uint32_t dst = buffers.getOrAssign(matmulOp.getDst());
      payloadType = fb::ExecutePayload_Matmul;
      payloadOffset = fb::CreateMatmul(builder, src0, src1, dst).Union();
    } else if (auto loadOp = dyn_cast<LoadOp>(op)) {
      uint32_t dst = buffers.getOrAssign(loadOp.getDst());
      payloadType = fb::ExecutePayload_LoadOp;
      payloadOffset = fb::CreateLoadOp(builder, dst).Union();
    } else if (auto storeOp = dyn_cast<StoreOp>(op)) {
      uint32_t src = buffers.getOrAssign(storeOp.getSrc());
      payloadType = fb::ExecutePayload_StoreOp;
      payloadOffset = fb::CreateStoreOp(builder, src).Union();
    }
  }

  return fb::CreateExecuteDirect(builder, &acquires, &semRequires, payloadType,
                                 payloadOffset);
}

/// Translate the full module into a FlatBuffer binary written to `os`.
static LogicalResult translateToFlatbuffer(ModuleOp module,
                                           llvm::raw_ostream &os) {
  flatbuffers::FlatBufferBuilder builder(4096);

  std::vector<flatbuffers::Offset<fb::Function>> functions;
  uint16_t maxSemAddr = 0;

  for (auto funcOp : module.getOps<func::FuncOp>()) {
    std::vector<flatbuffers::Offset<fb::Operation>> ops;
    BufferTable buffers;

    // Walk through operations in program order.
    funcOp.walk([&](Operation *op) {
      if (auto semAlloc = dyn_cast<SemAllocOp>(op)) {
        uint16_t addr = getSemAddress(semAlloc.getSemaphore());
        if (addr + 1 > maxSemAddr)
          maxSemAddr = addr + 1;
        uint32_t emptyCnt = getConstantIndex(semAlloc.getEmptyCount());
        uint32_t fullCnt = getConstantIndex(semAlloc.getFullCount());
        auto chainAddrs = getChainAddresses(semAlloc.getChainsFrom());
        uint16_t generation = getSemGeneration(semAlloc.getSemaphore());
        auto sa = fb::CreateSemAllocDirect(builder, addr, emptyCnt, fullCnt,
                                           &chainAddrs, generation);
        ops.push_back(fb::CreateOperation(builder, fb::Command_SemAlloc,
                                          sa.Union()));
      // SemDealloc is intentionally not emitted: with fresh-address-first
      // allocation, a later SemAlloc on a recycled address reinitializes
      // the semaphore, so the fabric needs no explicit release signal.
      // } else if (auto semDealloc = dyn_cast<SemDeallocOp>(op)) {
      //   uint16_t addr = getSemAddress(semDealloc.getSemaphore());
      //   auto sd = fb::CreateSemDealloc(builder, addr);
      //   ops.push_back(fb::CreateOperation(builder, fb::Command_SemDealloc,
      //                                     sd.Union()));
      } else if (auto execOp = dyn_cast<ExecuteOp>(op)) {
        auto exec = serializeExecute(builder, execOp, buffers);
        ops.push_back(fb::CreateOperation(builder, fb::Command_Execute,
                                          exec.Union()));
      } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        uint32_t bufId = buffers.getOrAssign(deallocOp.getMemref());
        auto md = fb::CreateMemDealloc(builder, bufId);
        ops.push_back(
            fb::CreateOperation(builder, fb::Command_MemDealloc, md.Union()));
      } else if (op->getParentOp() == funcOp) {
        // Emit MemAlloc for any top-level op that produces a memref result.
        for (Value result : op->getResults()) {
          if (isa<MemRefType>(result.getType())) {
            uint32_t bufId = buffers.getOrAssign(result);
            auto ma = fb::CreateMemAlloc(builder, bufId);
            ops.push_back(fb::CreateOperation(builder, fb::Command_MemAlloc,
                                              ma.Union()));
          }
        }
      }
    });

    // Materialize Function.buffers in declaration order.
    std::vector<flatbuffers::Offset<fb::BufferRef>> bufferRefs;
    bufferRefs.reserve(buffers.order.size());
    for (Value v : buffers.order)
      bufferRefs.push_back(buildBufferRef(builder, v));

    auto name = builder.CreateString(funcOp.getName().str());
    auto buffersVec = builder.CreateVector(bufferRefs);
    auto opsVec = builder.CreateVector(ops);
    functions.push_back(fb::CreateFunction(builder, name, buffersVec, opsVec));
  }

  // Serialize memref.global constants.
  std::vector<flatbuffers::Offset<fb::Constant>> constants;
  for (auto globalOp : module.getOps<memref::GlobalOp>()) {
    auto memrefType = globalOp.getType();
    std::string name = globalOp.getSymName().str();

    std::vector<uint32_t> shape;
    for (int64_t dim : memrefType.getShape())
      shape.push_back(static_cast<uint32_t>(dim));

    auto elemType = convertElementType(memrefType.getElementType());

    // Extract raw data bytes from the initial_value attribute.
    std::vector<uint8_t> data;
    if (auto initValue = globalOp.getInitialValue()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(*initValue)) {
        auto rawData = denseAttr.getRawData();
        if (denseAttr.isSplat()) {
          // For splats, getRawData() returns one element's bytes; replicate
          // across all elements so downstream consumers see the full payload.
          size_t numElems = denseAttr.getNumElements();
          data.reserve(rawData.size() * numElems);
          for (size_t i = 0; i < numElems; ++i)
            data.insert(data.end(), rawData.begin(), rawData.end());
        } else {
          data.assign(rawData.begin(), rawData.end());
        }
      }
    }

    constants.push_back(fb::CreateConstantDirect(builder, name.c_str(), &shape,
                                                 elemType, &data));
  }

  // Memory per tier — placeholder; could be computed from alloc attributes.
  // Index 0=Local SRAM, 1=Global SRAM, 2=DRAM.
  std::vector<uint64_t> memPerTier = {0, 0, 0};

  auto program = fb::CreateProgramDirect(builder, &functions, maxSemAddr,
                                         &memPerTier, &constants);
  builder.Finish(program);

  os.write(reinterpret_cast<const char *>(builder.GetBufferPointer()),
           builder.GetSize());
  return success();
}

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void mlir::eaac::registerEAACToFlatbufferTranslation(
    DialectRegistry &registry) {
  TranslateFromMLIRRegistration registration(
      "eaac-to-flatbuffer", "Serialize EAAC MLIR to FlatBuffer binary",
      [](ModuleOp module, llvm::raw_ostream &os) {
        return translateToFlatbuffer(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<eaac::EAACDialect, DLTIDialect, func::FuncDialect,
                        arith::ArithDialect, memref::MemRefDialect>();
      });
}
