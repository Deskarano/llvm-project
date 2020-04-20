#ifndef LLVM_MLIR_BLOCK_DIALECT_H
#define LLVM_MLIR_BLOCK_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Function.h"

namespace mlir {
namespace block {

class BlockDialect : public mlir::Dialect {
public:
  explicit BlockDialect(mlir::MLIRContext *ctx);

  static llvm::StringRef getDialectNamespace() { return "block"; }
};

#define GET_OP_CLASSES
#include "block/Ops.h.inc"

} // namespace block
} // namespace mlir

#endif //LLVM_MLIR_BLOCK_DIALECT_H
