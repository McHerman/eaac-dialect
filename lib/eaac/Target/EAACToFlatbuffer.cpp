//===- EAACToFlatbuffer.cpp - EAAC to FlatBuffer translation --------------===//
//
// Translates a fully-lowered EAAC MLIR module into a FlatBuffer binary.
//
//===----------------------------------------------------------------------===//

#include "eaac/Target/EAACToFlatbuffer.h"
#include "eaac/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include "flatbuffers/flatbuffers.h"
#include "eaac_program_generated.h"

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

/// Build a BufferRef from a memref Value, reading eaac.offset and eaac.tier
/// from its defining memref.alloc op.
static flatbuffers::Offset<fb::BufferRef>
createBufferRef(flatbuffers::FlatBufferBuilder &builder, Value memref) {
  auto memrefType = cast<MemRefType>(memref.getType());

  uint64_t offset = 0;
  uint8_t tier = 0;

  // Walk through sem_require / sem_acquire to find the underlying alloc.
  Value current = memref;
  while (current) {
    if (auto allocOp = current.getDefiningOp<memref::AllocOp>()) {
      if (auto attr = allocOp->getAttrOfType<IntegerAttr>("eaac.offset"))
        offset = attr.getInt();
      if (auto attr = allocOp->getAttrOfType<IntegerAttr>("eaac.tier"))
        tier = attr.getInt();
      break;
    }
    if (auto reqOp = current.getDefiningOp<SemRequireOp>()) {
      current = reqOp.getMemref();
      continue;
    }
    if (auto acqOp = current.getDefiningOp<SemAcquireOp>()) {
      current = acqOp.getMemref();
      continue;
    }
    if (auto getGlobalOp = current.getDefiningOp<memref::GetGlobalOp>()) {
      // Global memrefs have no dynamic offset/tier — use defaults.
      break;
    }
    // Function argument or unknown — no offset/tier info.
    break;
  }

  std::vector<uint32_t> shape;
  for (int64_t dim : memrefType.getShape())
    shape.push_back(static_cast<uint32_t>(dim));

  return fb::CreateBufferRefDirect(builder, offset, tier,
                                   &shape,
                                   convertElementType(memrefType.getElementType()));
}

/// Get the hardware semaphore address from a sem_alloc op via eaac.sem_addr.
static uint16_t getSemAddress(Value semaphore) {
  auto allocOp = semaphore.getDefiningOp<SemAllocOp>();
  if (!allocOp)
    return 0;
  if (auto attr = allocOp->getAttrOfType<IntegerAttr>("eaac.sem_addr"))
    return attr.getInt();
  return 0;
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
serializeExecute(flatbuffers::FlatBufferBuilder &builder, ExecuteOp execOp) {
  std::vector<flatbuffers::Offset<fb::SemDep>> acquires;
  std::vector<flatbuffers::Offset<fb::SemDep>> semRequires;
  flatbuffers::Offset<void> payloadOffset;
  fb::ExecutePayload payloadType = fb::ExecutePayload_NONE;

  for (Operation &op : execOp.getBody().front()) {
    if (auto acqOp = dyn_cast<SemAcquireOp>(op)) {
      auto bufRef = createBufferRef(builder, acqOp.getMemref());
      acquires.push_back(
          fb::CreateSemDep(builder, getSemAddress(acqOp.getSemaphore()),
                           bufRef));
    } else if (auto reqOp = dyn_cast<SemRequireOp>(op)) {
      auto bufRef = createBufferRef(builder, reqOp.getMemref());
      semRequires.push_back(
          fb::CreateSemDep(builder, getSemAddress(reqOp.getSemaphore()),
                           bufRef));
    } else if (auto dmaOp = dyn_cast<DmaStartOp>(op)) {
      auto src = createBufferRef(builder, dmaOp.getSrc());
      auto dst = createBufferRef(builder, dmaOp.getDst());
      payloadType = fb::ExecutePayload_DmaStart;
      payloadOffset = fb::CreateDmaStart(builder, src, dst).Union();
    } else if (auto matmulOp = dyn_cast<MatmulOp>(op)) {
      std::string opName = op.getName().getStringRef().str();
      std::vector<flatbuffers::Offset<fb::BufferRef>> inputs;
      std::vector<flatbuffers::Offset<fb::BufferRef>> outputs;
      inputs.push_back(createBufferRef(builder, matmulOp.getSrc0()));
      inputs.push_back(createBufferRef(builder, matmulOp.getSrc1()));
      outputs.push_back(createBufferRef(builder, matmulOp.getDst()));
      payloadType = fb::ExecutePayload_Compute;
      payloadOffset =
          fb::CreateComputeDirect(builder, opName.c_str(), &inputs, &outputs)
              .Union();
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
    std::vector<flatbuffers::Offset<fb::BufferRef>> args;
    std::vector<flatbuffers::Offset<fb::BufferRef>> results;

    // Serialize function arguments.
    for (BlockArgument arg : funcOp.getArguments()) {
      if (isa<MemRefType>(arg.getType()))
        args.push_back(createBufferRef(builder, arg));
    }

    // Walk through operations in program order.
    funcOp.walk([&](Operation *op) {
      if (auto semAlloc = dyn_cast<SemAllocOp>(op)) {
        uint16_t addr = getSemAddress(semAlloc.getSemaphore());
        if (addr + 1 > maxSemAddr)
          maxSemAddr = addr + 1;
        uint32_t emptyCnt = getConstantIndex(semAlloc.getEmptyCount());
        uint32_t fullCnt = getConstantIndex(semAlloc.getFullCount());
        auto sa = fb::CreateSemAlloc(builder, addr, emptyCnt, fullCnt);
        ops.push_back(fb::CreateOperation(builder, fb::Command_SemAlloc,
                                          sa.Union()));
      } else if (auto semDealloc = dyn_cast<SemDeallocOp>(op)) {
        uint16_t addr = getSemAddress(semDealloc.getSemaphore());
        auto sd = fb::CreateSemDealloc(builder, addr);
        ops.push_back(fb::CreateOperation(builder, fb::Command_SemDealloc,
                                          sd.Union()));
      } else if (auto execOp = dyn_cast<ExecuteOp>(op)) {
        auto exec = serializeExecute(builder, execOp);
        ops.push_back(fb::CreateOperation(builder, fb::Command_Execute,
                                          exec.Union()));
      } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
        auto dst = createBufferRef(builder, fillOp.getDpsInits()[0]);
        // Extract fill value as raw bytes.
        std::vector<uint8_t> valueBytes;
        Value fillVal = fillOp.getDpsInputs()[0];
        if (auto constOp = fillVal.getDefiningOp<arith::ConstantOp>()) {
          if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
            int64_t v = intAttr.getInt();
            unsigned bitWidth = intAttr.getType().getIntOrFloatBitWidth();
            unsigned byteWidth = (bitWidth + 7) / 8;
            for (unsigned i = 0; i < byteWidth; ++i)
              valueBytes.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
          } else if (auto fpAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
            APFloat apf = fpAttr.getValue();
            APInt bits = apf.bitcastToAPInt();
            unsigned byteWidth = (bits.getBitWidth() + 7) / 8;
            for (unsigned i = 0; i < byteWidth; ++i)
              valueBytes.push_back(
                  static_cast<uint8_t>(bits.extractBitsAsZExtValue(8, i * 8)));
          }
        }
        auto fill = fb::CreateFillDirect(builder, dst, &valueBytes);
        ops.push_back(
            fb::CreateOperation(builder, fb::Command_Fill, fill.Union()));
      } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        auto buf = createBufferRef(builder, allocOp.getResult());
        auto ma = fb::CreateMemAlloc(builder, buf);
        ops.push_back(
            fb::CreateOperation(builder, fb::Command_MemAlloc, ma.Union()));
      } else if (auto deallocOp = dyn_cast<memref::DeallocOp>(op)) {
        auto buf = createBufferRef(builder, deallocOp.getMemref());
        auto md = fb::CreateMemDealloc(builder, buf);
        ops.push_back(
            fb::CreateOperation(builder, fb::Command_MemDealloc, md.Union()));
      }
    });

    // Serialize return values.
    if (auto returnOp = funcOp.getBody().front().getTerminator()) {
      for (Value operand : returnOp->getOperands()) {
        if (isa<MemRefType>(operand.getType()))
          results.push_back(createBufferRef(builder, operand));
      }
    }

    auto name = builder.CreateString(funcOp.getName().str());
    auto argsVec = builder.CreateVector(args);
    auto resultsVec = builder.CreateVector(results);
    auto opsVec = builder.CreateVector(ops);
    functions.push_back(
        fb::CreateFunction(builder, name, argsVec, resultsVec, opsVec));
  }

  // Memory per tier — placeholder; could be computed from alloc attributes.
  std::vector<uint64_t> memPerTier = {0, 0, 0, 0}; // indices 0-3

  auto program = fb::CreateProgramDirect(builder, &functions, maxSemAddr,
                                         &memPerTier);
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
        registry.insert<eaac::EAACDialect, func::FuncDialect,
                        arith::ArithDialect, memref::MemRefDialect,
                        linalg::LinalgDialect>();
      });
}
