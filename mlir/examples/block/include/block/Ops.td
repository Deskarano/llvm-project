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

        OpBuilder<"Builder *builder, OperationState &state, int64_t value">
    ];

    let verifier = [{ return ::verify(*this); }];
}

def ConstantBooleanOp : Block_Op<"const_bool", [NoSideEffect]> {
    let summary = "constant boolean";

    let arguments = (ins BoolAttr:$value);
    let results = (outs BoolLike);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                    "BoolAttr value", [{
            build(builder, state, value.getType(), value);
        }]>,

        OpBuilder<"Builder *builder, OperationState &state, bool value">
    ];
}

class Binary_Op<string mnemonic> : Block_Op<mnemonic> {
    let arguments = (ins I64:$lhs, I64:$rhs);
    let results = (outs I64);

    let builders = [
        OpBuilder<"Builder *b, OperationState &state, Value lhs, Value rhs">
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

class Boolean_Op<string mnemonic> : Block_Op<mnemonic> {
    let arguments = (ins BoolLike:$lhs, BoolLike:$rhs);
    let results = (outs BoolLike);

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
    let results = (outs BoolLike);

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
#endif // BLOCK_OPS