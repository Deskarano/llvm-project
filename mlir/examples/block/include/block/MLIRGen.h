#ifndef LLVM_MLIR_BLOCK_MLIRGEN_H
#define LLVM_MLIR_BLOCK_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
class OwningModuleRef;
} // namespace mlir

namespace block {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context, ModuleAST &moduleAST);
} // namespace toy

#endif //LLVM_MLIR_BLOCK_MLIRGEN_H
