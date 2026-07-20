//===- EAAC.td - EAAC dialect definition -------------------*- tablegen -*-===//
//
// Defines the EAAC dialect.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_DIALECT
#define EAAC_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// EAAC Dialect
//===----------------------------------------------------------------------===//

def EAAC_Dialect : Dialect {
  let name = "eaac";
  let summary = "EAAC example dialect";
  let cppNamespace = "::mlir::eaac";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base class for EAAC operations
//===----------------------------------------------------------------------===//

class EAAC_Op<string mnemonic, list<Trait> traits = []> :
    Op<EAAC_Dialect, mnemonic, traits>;

#endif // EAAC_DIALECT
