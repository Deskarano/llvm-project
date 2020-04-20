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
    llvm::outs() << "declaring " << var << " with value" << value << "\n";

    if (symbolTable.count(var))
      return mlir::failure();
    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::FuncOp mlirGen(BlockAST &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);

    std::string blockName = blockAST.getProto()->getName();
    auto location = loc(blockAST.getProto()->loc());

    std::unique_ptr<PropertyAST> input, output, state;
    bool inputFound, outputFound, stateFound;
    inputFound = outputFound = stateFound = false;

    for (auto &p : *blockAST.getProperties()) {
      if (p->getKind() == "input") {
        if (inputFound)
          return nullptr; // duplicate inputs
        else {
          inputFound = true;
          input = std::move(p);
        }
      } else if (p->getKind() == "output") {
        if (outputFound)
          return nullptr; // duplicate outputs
        else {
          outputFound = true;
          output = std::move(p);
        }
      } else if (p->getKind() == "state") {
        if (stateFound)
          return nullptr; // duplicate state
        else {
          stateFound = true;
          state = std::move(p);
        }
      }
    }

    // create the function type
    std::vector<mlir::Type> inputTypes, outputTypes;

    if(inputFound)
    {
      for (auto &var : *input->getVars())
        inputTypes.push_back(mlir::IntegerType::get(var->getBounds()->getUpper() -
                                                        var->getBounds()->getLower() + 1,
                                                    builder.getContext()));
    }

    if(outputFound)
    {
      for (auto &var : *output->getVars())
        outputTypes.push_back(mlir::IntegerType::get(var->getBounds()->getUpper() -
                                                         var->getBounds()->getLower() + 1,
                                                     builder.getContext()));
    }

    // create the function itself
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);
    auto function = mlir::FuncOp::create(location, blockName, funcType);
    auto &entryBlock = *function.addEntryBlock();

    if(inputFound)
    {
      for (const auto &nameValue : llvm::zip(*input->getVars(),
                                             entryBlock.getArguments())) {
        if (failed(declare(std::get<0>(nameValue)->getName(),
                           std::get<1>(nameValue))))
          return nullptr;
      }
    }

    // push declared variables
    if(outputFound)
    {
      for(const auto &nameValue : llvm::zip(*output->getVars(),
                                              outputTypes))
        if(failed(declare(std::get<0>(nameValue)->getName(), mlir::Value(0))
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