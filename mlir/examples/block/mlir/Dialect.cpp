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

/// Constants

void ConstantNumberOp::build(mlir::Builder *builder,
                             mlir::OperationState &state,
                             int64_t value) {
  auto dataType = IntegerType::get(64, builder->getContext());
  auto dataAttribute = IntegerAttr::get(dataType, value);

  ConstantNumberOp::build(builder, state, dataType, dataAttribute);
}

static mlir::LogicalResult verify(ConstantNumberOp op) {
  return success();
}

void ConstantBooleanOp::build(mlir::Builder *builder,
                              mlir::OperationState &state,
                              bool value) {
  auto dataType = IntegerType::get(1, builder->getContext());
  auto dataAttribute = IntegerAttr::get(dataType, value ? 1 : 0);

  ConstantBooleanOp::build(builder, state, dataType, dataAttribute);
}

static mlir::LogicalResult verify(ConstantBooleanOp op) {
  return success();
}

/// Arithmetic Operations

void AddOp::build(Builder *b, OperationState &state,
                  Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(64, b->getContext()));
  state.addOperands({lhs, rhs});
}

void SubOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(64, b->getContext()));
  state.addOperands({lhs, rhs});
}

void BitwiseAndOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(64, b->getContext()));
  state.addOperands({lhs, rhs});
}

void BitwiseOrOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(64, b->getContext()));
  state.addOperands({lhs, rhs});
}
void BitwiseXorOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(64, b->getContext()));
  state.addOperands({lhs, rhs});
}

/// Boolean Operations

void BooleanAndOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void BooleanOrOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

/// Comparison Operations

void LessThanOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void LessThanOrEqualOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void GreaterThanOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void GreaterThanOrEqualOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void EqualOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

void NotEqualOp::build(Builder *b, OperationState &state, Value lhs, Value rhs) {
  state.addTypes(IntegerType::get(1, b->getContext()));
  state.addOperands({lhs, rhs});
}

#define GET_OP_CLASSES
#include "block/Ops.cpp.inc"