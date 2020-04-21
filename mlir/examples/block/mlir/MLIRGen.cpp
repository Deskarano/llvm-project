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

  mlir::ModuleOp mlirGen(ModuleAST &moduleAST) {
    theModule = mlir::ModuleOp::create(builder.getUnknownLoc());

    for (BlockAST &b : moduleAST) {
      auto block = mlirGen(b);
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

  mlir::Value mlirGen(ExprAST *expr) {
    switch(expr->getKind()) {
    case block::ExprAST::Expr_Const:
      return nullptr;

    case block::ExprAST::Expr_Var:
      return nullptr;

    case block::ExprAST::Expr_BinOp:
      return nullptr;

    case block::ExprAST::Expr_Cond:
      return nullptr;

    case block::ExprAST::Expr_Assign:
      return nullptr;
    }
  }

  mlir::LogicalResult mlirGen(ExprASTList *exprList) {

  }

  mlir::LogicalResult mlirGen(EventASTList *events) {
    // parse all the "always" events first - these go at the beginning of the function block
    for (auto &e : *events) {
      if (e->getKind() == "always") {
        if (e->getCondition() != nullptr) {
          emitError(loc(e->loc()), "always event with condition");
          return mlir::failure();
        }

        if (failed(mlirGen(e->getAction())))
          return mlir::failure();
      }
    }

    // then parse the "when" events
    for (auto &e : *events) {
      if (e->getKind() == "when") {
        if(e->getCondition() == nullptr)
        {
          emitError(loc(e->loc()), "when event without condition");
          return mlir::failure();
        }

        if(failed(mlirGen(e->getCondition())))
          return mlir::failure();

        if(failed(mlirGen(e->getAction())))
          return mlir::failure();
      }
    }
  }

  mlir::FuncOp mlirGen(PrototypeAST *proto, PropertyASTList *properties) {
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

  mlir::FuncOp mlirGen(BlockAST &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
    mlir::FuncOp function(mlirGen(blockAST.getProto(),
                                  blockAST.getProperties()));

    if (failed(mlirGen(blockAST.getEvents()))) {
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
  return MLIRGenImpl(context).mlirGen(moduleAST);
}

} // namespace block