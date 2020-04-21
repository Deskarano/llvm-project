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

def ConstantOp : Block_Op<"constant", [NoSideEffect]> {
    let summary = "constant";

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

class Binary_Op<string mnemonic> : Block_Op<mnemonic> {
    let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
    let results = (outs AnyInteger);

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

def AndOp : Binary_Op<"and"> {
    let summary = "bitwise and operator";
}

def OrOp : Binary_Op<"or"> {
    let summary = "bitwise or operator";
}

def XorOp : Binary_Op<"xor"> {
    let summary = "bitwise exlusive or operator";
}

#endif // BLOCK_OPS