//===- EAACToFlatbuffer.h - EAAC to FlatBuffer translation ------*- C++ -*-===//
//
// Declares the translation from EAAC MLIR to FlatBuffer binary format.
//
//===----------------------------------------------------------------------===//

#ifndef EAAC_TARGET_EAACTOFLATBUFFER_H
#define EAAC_TARGET_EAACTOFLATBUFFER_H

namespace mlir {
class DialectRegistry;

namespace eaac {

/// Register the EAAC-to-FlatBuffer translation with mlir-translate.
void registerEAACToFlatbufferTranslation(DialectRegistry &registry);

} // namespace eaac
} // namespace mlir

#endif // EAAC_TARGET_EAACTOFLATBUFFER_H
