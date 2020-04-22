#include "block/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/StandardTypes.h"

using namespace mlir;
using namespace mlir::block;

BlockDialect::BlockDialect(mlir::MLIRContext *ctx) : mlir::Dialect("block", ctx) {
  addOperations<
#define GET_OP_LIST
#include "block/Ops.cpp.inc"
  >();
}

void ConstantNumberOp::build(mlir::Builder *builder,
                       mlir::OperationState &state,
                       int64_t value) {
  auto dataType = IntegerType::get(64, builder->getContext());
  auto dataAttribute = IntegerAttr::get(dataType, value);

  ConstantNumberOp::build(builder, state, dataType, dataAttribute);
}

static mlir::LogicalResult verify(ConstantNumberOp op)
{
  return success();
}
//
//void ConstantBooleanOp::build(mlir::Builder *builder,
//                             mlir::OperationState &state,
//                             bool value) {
//  auto dataType = nullptr;
//  auto dataAttribute = BoolAttr::get(value, builder->getContext());
//
//  ConstantBooleanOp::build(builder, state, dataType, dataAttribute);
//}
//
//static mlir::LogicalResult verify(ConstantBooleanOp op)
//{
//  return success();
//}

#define GET_OP_CLASSES
#include "block/Ops.cpp.inc"