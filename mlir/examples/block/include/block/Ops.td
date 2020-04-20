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
    let results = (outs I64);

    let builders = [
        OpBuilder<"Builder *builder, OperationState &state, "
                  "IntegerAttr value", [{
            build(builder, state, value.getType(), value);
        }]>,

        OpBuilder<"Builder *builder, OperationState &state, int64_t value">
    ];

    let verifier = [{ return ::verify(*this); }];
}

#endif // BLOCK_OPS