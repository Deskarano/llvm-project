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
                             int64_t value, int size) {
  auto dataType = IntegerType::get(size, builder->getContext());
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

void ConstantEmptyOp::build(Builder *builder, OperationState &state, llvm::StringRef value, int size) {
  auto resultType = IntegerType::get(size, builder->getContext());
  auto attribute = StringAttr::get(value, builder->getContext());

  ConstantEmptyOp::build(builder, state, resultType, attribute);
}

void SliceOp::build(Builder *b, OperationState &state, Value val, int upper, int lower) {
  auto dataType = IntegerType::get(abs(upper - lower) + 1, b->getContext());
  auto sliceType = IntegerType::get(64, b->getContext());;
  SliceOp::build(b, state, dataType, val,
                 IntegerAttr::get(sliceType, upper),
                 IntegerAttr::get(sliceType, lower));
}

/// Arithmetic Operations

void AddOp::build(Builder *b, OperationState &state,
                  Value lhs, Value rhs, int size) {
  state.addTypes(IntegerType::get(size, b->getContext()));
  state.addOperands({lhs, rhs});
}

void SubOp::build(Builder *b, OperationState &state,
                  Value lhs, Value rhs, int size) {
  state.addTypes(IntegerType::get(size, b->getContext()));
  state.addOperands({lhs, rhs});
}

void BitwiseAndOp::build(Builder *b, OperationState &state,
                         Value lhs, Value rhs, int size) {
  state.addTypes(IntegerType::get(size, b->getContext()));
  state.addOperands({lhs, rhs});
}

void BitwiseOrOp::build(Builder *b, OperationState &state,
                        Value lhs, Value rhs, int size) {
  state.addTypes(IntegerType::get(size, b->getContext()));
  state.addOperands({lhs, rhs});
}
void BitwiseXorOp::build(Builder *b, OperationState &state,
                         Value lhs, Value rhs, int size) {
  state.addTypes(IntegerType::get(size, b->getContext()));
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

/// Function operators

void EventCall::build(Builder *builder, OperationState &state, Value condition, Block *dst) {
  state.addOperands(condition);
  state.addSuccessor(dst, mlir::ValueRange());
}

void MergeOp::build(Builder *builder, OperationState &state, ArrayRef<Value> arguments) {
  int sum = 0;
  for(auto &a : arguments) {
    sum += a.getType().getIntOrFloatBitWidth();
  }

  state.addOperands(arguments);
  state.addTypes(IntegerType::get(sum, builder->getContext()));
}

void ReturnOp::build(Builder *builder, OperationState &state, ArrayRef<Value> arguments) {
  state.addOperands(arguments);
}

#define GET_OP_CLASSES
#include "block/Ops.cpp.inc"