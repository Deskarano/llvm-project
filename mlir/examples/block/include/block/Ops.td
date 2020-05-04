#ifndef BLOCK_OPS
#define BLOCK_OPS

include "mlir/IR/OpBase.td"

def Block_Dialect : Dialect {
    let name = "block";
    let cppNamespace = "block";
}

class Block_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Block_Dialect, mnemonic, traits>;

/// Operations

def ConstantNumberOp : Block_Op<"const_num", [NoSideEffect]> {
    let summary = "constant number";

    let arguments = (ins I64Attr:$value);
    let results = (outs AnyInteger);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                  "IntegerAttr value", [{
            build(builder, state, value.getType(), value);
        }]>,

        OpBuilder<"Builder *builder, OperationState &state, int value, int size">
    ];

    let verifier = [{ return ::verify(*this); }];
}

def ConstantBooleanOp : Block_Op<"const_bool", [NoSideEffect]> {
    let summary = "constant boolean";

    let arguments = (ins I1Attr:$value);
    let results = (outs I1);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                    "IntegerAttr value", [{
            build(builder, state, value.getType(), value);
        }]>,

        OpBuilder<"Builder *builder, OperationState &state, bool value">
    ];
}

def ConstantEmptyOp : Block_Op<"const_empty", [NoSideEffect]> {
    let summary = "constant empty";

    let arguments = (ins StrAttr:$value);
    let results = (outs AnyInteger);

    let builders = [
        OpBuilder<"Builder *builder, OperationState *state, "
                    "StringAttr value", [{
            build(builder, state, value);
        }]>,

        OpBuilder<"Builder *builder, OperationState &state, llvm::StringRef value, int size">
    ];
}

def SliceOp : Block_Op<"slice"> {
    let summary = "variable slice operator";

    let arguments = (ins AnyInteger:$val, I64Attr:$upper, I64Attr:$lower);
    let results = (outs AnyInteger);

    let builders = [
        OpBuilder<"Builder *b, OperationState &state, Value val, int upper, int lower">
    ];
}

class Binary_Op<string mnemonic> : Block_Op<mnemonic> {
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger);

    let builders = [
        OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs, int size">
    ];
}

def AddOp : Binary_Op<"add"> {
    let summary = "addition operator";
}

def SubOp : Binary_Op<"sub"> {
    let summary = "subtraction operator";
}

def BitwiseAndOp : Binary_Op<"bit_and"> {
    let summary = "bitwise and operator";
}

def BitwiseOrOp : Binary_Op<"bit_or"> {
    let summary = "bitwise or operator";
}

def BitwiseXorOp : Binary_Op<"bit_xor"> {
    let summary = "bitwise exlusive or operator";
}

class Boolean_Op<string mnemonic> : Block_Op<mnemonic, [NativeOpTrait<"ResultsAreBoolLike">]> {
    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1);

    let builders = [
        OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
    ];
}

def BooleanAndOp : Boolean_Op<"bool_and"> {
    let summary = "boolean and operator";
}

def BooleanOrOp : Boolean_Op<"bool_or"> {
    let summary = "boolean or operator";
}

class Comparison_Op<string mnemonic> : Block_Op<mnemonic> {
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs I1);

    let builders = [
        OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
    ];
}

def LessThanOp : Comparison_Op<"lt"> {
    let summary = "less than operator";
}

def LessThanOrEqualOp : Comparison_Op<"lte"> {
    let summary = "less than or equal to operator";
}

def GreaterThanOp : Comparison_Op<"gt"> {
    let summary = "greater than operator";
}

def GreaterThanOrEqualOp : Comparison_Op<"gte"> {
    let summary = "greater than or equal to operator";
}

def EqualOp : Comparison_Op<"eq"> {
    let summary = "equal to operator";
}

def NotEqualOp : Comparison_Op<"neq"> {
    let summary = "not equal to operator";
}

/// Control flow

def EventCall : Block_Op<"eventcall", [Terminator]> {
    let summary = "event call operator";

    let arguments = (ins I1:$condition);
    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                  "Value condition, Block *trueBlock, Block *falseBlock">
    ];
}

def EventDone : Block_Op<"eventdone", [Terminator]> {
    let summary = "event finish operator";

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, Block *next">
    ];
}

def MergeOp : Block_Op<"merge"> {
    let summary = "output merge operator";

    let arguments = (ins Variadic<AnyInteger>:$inputs);
    let results = (outs AnyInteger:$output);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                    "ArrayRef<Value> arguments">
    ];
}

def ReturnOp : Block_Op<"return", [Terminator, HasParent<"FuncOp">]> {
    let summary = "return operator";

    let arguments = (ins Variadic<AnyInteger>:$input);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                    "ArrayRef<Value> arguments">
    ];
}

#endif // BLOCK_OPS