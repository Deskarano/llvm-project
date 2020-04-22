#include "block/MLIRGen.h"
#include "block/AST.h"
#include "block/Dialect.h"

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace mlir::block;
using namespace block;

namespace {

class MLIRGenImpl {
public:
  MLIRGenImpl(mlir::MLIRContext &context) : builder(&context) {}

  mlir::ModuleOp mlirGenModule(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (BlockAST &b : moduleAST) {
      auto block = mlirGenBlock(b);
      if (!block)
        return nullptr;

      theModule.push_back(block);
    }

//    if (failed(mlir::verify(theModule))) {
//      theModule.emitError("module verification error");
//      return nullptr;
//    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;

  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file),
                                     loc.line, loc.col);
  }

  mlir::LogicalResult declare(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::Value mlirGenBitsConstExpr(BitsConstExprAST &expr) {
    return builder.create<ConstantNumberOp>(loc(expr.loc()), expr.getValue());
  }

  mlir::Value mlirGenBoolConstExpr(BoolConstExprAST &expr) {
    return builder.create<ConstantBooleanOp>(loc(expr.loc()), expr.getValue());
  }

  mlir::Value mlirGenBitsVarExpr(BitsVarExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName()))
      return variable;

    emitError(loc(expr.loc()), "unknown variable '")
        << expr.getName() << "'";
    return nullptr;
  }

  mlir::Value mlirGenBitsBinaryExpr(BitsBinaryExprAST &expr) {
    mlir::Value lhs = mlirGenBitsExpr(*expr.getLHS());
    if (!lhs)
      return nullptr;

    mlir::Value rhs = mlirGenBitsExpr(*expr.getRHS());
    if (!rhs)
      return nullptr;

    auto location = loc(expr.loc());
    if (expr.getOp() == "+")
      return builder.create<AddOp>(location,  lhs, rhs);

    else if (expr.getOp() == "-")
      return builder.create<SubOp>(location, lhs, rhs);

    else if (expr.getOp() == "&")
      return builder.create<BitwiseAndOp>(location, lhs, rhs);

    else if (expr.getOp() == "|")
      return builder.create<BitwiseOrOp>(location, lhs, rhs);

    else if (expr.getOp() == "^")
      return builder.create<BitwiseXorOp>(location, lhs, rhs);

    else {
      emitError(location, "unknown operation '") << expr.getOp() << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGenBoolBinaryExpr(BoolBinaryExprAST &expr) {
    mlir::Value lhs = mlirGenBoolExpr(*expr.getLHS());
    if (!lhs)
      return nullptr;

    mlir::Value rhs = mlirGenBoolExpr(*expr.getRHS());
    if (!rhs)
      return nullptr;

    auto location = loc(expr.loc());
    if (expr.getOp() == "&&")
      return builder.create<BooleanAndOp>(location, lhs, rhs);

    else if (expr.getOp() == "||")
      return builder.create<BooleanOrOp>(location, lhs, rhs);

    else {
      emitError(location, "unknown operation '") << expr.getOp() << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGenBitsAssignExpr(BitsAssignExprAST &expr) {

  }

  mlir::Value mlirGenCompExpr(BoolCompExprAST &expr) {
    mlir::Value lhs = mlirGenBitsExpr(*expr.getLHS());
    if (!lhs)
      return nullptr;

    mlir::Value rhs = mlirGenBitsExpr(*expr.getRHS());
    if (!rhs)
      return nullptr;

    auto location = loc(expr.loc());
    if (expr.getOp() == "<")
      return builder.create<LessThanOp>(location, lhs, rhs);

    else if (expr.getOp() == "<=")
      return builder.create<LessThanOrEqualOp>(location, lhs, rhs);

    else if (expr.getOp() == ">")
      return builder.create<GreaterThanOp>(location, lhs, rhs);

    else if (expr.getOp() == ">=")
      return builder.create<GreaterThanOrEqualOp>(location, lhs, rhs);

    else if (expr.getOp() == "==")
      return builder.create<EqualOp>(location, lhs, rhs);

    else if (expr.getOp() == "!=")
      return builder.create<NotEqualOp>(location, lhs, rhs);

    else {
      emitError(location, "unknown operation '") << expr.getOp() << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGenBitsExpr(BitsExprAST &expr) {
    switch (expr.getKind()) {
    case block::ExprAST::Expr_Const:
      return mlirGenBitsConstExpr(llvm::cast<BitsConstExprAST>(expr));

    case block::ExprAST::Expr_Var:
      return mlirGenBitsVarExpr(llvm::cast<BitsVarExprAST>(expr));

    case block::ExprAST::Expr_BinOp:
      return mlirGenBitsBinaryExpr(llvm::cast<BitsBinaryExprAST>(expr));

    case block::ExprAST::Expr_Special:
      return mlirGenBitsAssignExpr(llvm::cast<BitsAssignExprAST>(expr));

    default:
      emitError(loc(expr.loc()), "unknown expression type");
    }

    return nullptr;
  }

  mlir::Value mlirGenBoolExpr(BoolExprAST &expr) {
    switch (expr.getKind()) {
    case block::ExprAST::Expr_Const:
      return mlirGenBoolConstExpr(llvm::cast<BoolConstExprAST>(expr));

    case block::ExprAST::Expr_Var:
      emitError(loc(expr.loc()), "boolean variables are unsupported");
      return nullptr;

    case block::ExprAST::Expr_BinOp:
      return mlirGenBoolBinaryExpr(llvm::cast<BoolBinaryExprAST>(expr));

    case block::ExprAST::Expr_Special:
      return mlirGenCompExpr(llvm::cast<BoolCompExprAST>(expr));

    default:
      emitError(loc(expr.loc()), "unknown expression type");
    }

    return nullptr;
  }

  mlir::LogicalResult mlirGenActions(BitsExprASTList *exprList) {

  }

  mlir::LogicalResult mlirGenEvents(EventASTList *events) {
    // parse all the "always" events first - these go at the beginning of the function block
    for (auto &e : *events) {
      if (e->getKind() == "always") {
        if (e->getCondition() != nullptr) {
          emitError(loc(e->loc()), "always event with condition");
          return mlir::failure();
        }

        if (failed(mlirGenActions(e->getAction())))
          return mlir::failure();
      }
    }

    // then parse the "when" events
    for (auto &e : *events) {
      if (e->getKind() == "when") {
        if (e->getCondition() == nullptr) {
          emitError(loc(e->loc()), "when event without condition");
          return mlir::failure();
        }

        if (mlirGenBoolExpr(*e->getCondition()) == nullptr)
          return mlir::failure();

        if (failed(mlirGenActions(e->getAction())))
          return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::FuncOp mlirGenHeader(PrototypeAST *proto, PropertyASTList *properties) {
    std::string blockName = proto->getName();
    auto location = loc(proto->loc());

    std::unique_ptr<PropertyAST> input, output, state;
    bool inputFound, outputFound, stateFound;
    inputFound = outputFound = stateFound = false;

    for (auto &p : *properties) {
      if (p->getKind() == "input") {
        if (inputFound) {
          emitError(loc(p->loc()), "duplicate input declaration");
          return nullptr;
        } else {
          inputFound = true;
          input = std::move(p);
        }
      } else if (p->getKind() == "output") {
        if (outputFound) {
          emitError(loc(p->loc()), "duplicate output declaration");
          return nullptr;
        } else {
          outputFound = true;
          output = std::move(p);
        }
      } else if (p->getKind() == "state") {
        if (stateFound) {
          emitError(loc(p->loc()), "duplicate state declaration");
          return nullptr;
        } else {
          stateFound = true;
          state = std::move(p);
        }
      }
    }

    // create the function type
    std::vector<mlir::Type> inputTypes, outputTypes;

    if (inputFound) {
      for (auto &var : *input->getVars())
        inputTypes.push_back(mlir::IntegerType::get(var->getBounds()->getUpper() -
                                                        var->getBounds()->getLower() + 1,
                                                    builder.getContext()));
    }

    if (outputFound) {
      for (auto &var : *output->getVars())
        outputTypes.push_back(mlir::IntegerType::get(var->getBounds()->getUpper() -
                                                         var->getBounds()->getLower() + 1,
                                                     builder.getContext()));
    }

    // create the function itself
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);
    auto function = mlir::FuncOp::create(location, blockName, funcType);
    auto &entryBlock = *function.addEntryBlock();

    if (inputFound) {
      for (const auto &nameValue : llvm::zip(*input->getVars(),
                                             entryBlock.getArguments())) {
        if (failed(declare(std::get<0>(nameValue)->getName(),
                           std::get<1>(nameValue))))
          return nullptr;
      }
    }

    builder.setInsertionPointToStart(&entryBlock);

    // todo: push variables declared in state and output
    return function;
  }

  mlir::FuncOp mlirGenBlock(BlockAST &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
    mlir::FuncOp function(mlirGenHeader(blockAST.getProto(),
                                        blockAST.getProperties()));

    if (failed(mlirGenEvents(blockAST.getEvents()))) {
      function.erase();
      return nullptr;
    }

    return function;
  }
};

} // namespace

namespace block {

// The public API for codegen.
mlir::OwningModuleRef mlirGen(mlir::MLIRContext &context,
                              ModuleAST &moduleAST) {
  return MLIRGenImpl(context).mlirGenModule(moduleAST);
}

} // namespace block