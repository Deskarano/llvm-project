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

    theModule.dump();

    if (failed(mlir::verify(theModule))) {
      theModule.emitError("module verification error");
      return nullptr;
    }

    return theModule;
  }

private:
  mlir::ModuleOp theModule;
  mlir::OpBuilder builder;

  using outputValueList = std::vector<std::tuple<VarBoundsAST *, mlir::Value>>;

  llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
  llvm::ScopedHashTable<llvm::StringRef, VarBoundsAST *> boundsTable;
  llvm::ScopedHashTable<llvm::StringRef, outputValueList *> outputTable;
  llvm::ScopedHashTable<llvm::StringRef, std::vector<VarBoundsAST> *> availableOutputTable;

  std::vector<llvm::StringRef> inputNames, outputNames, stateNames;

  mlir::Location loc(Location loc) {
    return builder.getFileLineColLoc(builder.getIdentifier(*loc.file),
                                     loc.line, loc.col);
  }

  mlir::LogicalResult declareInput(llvm::StringRef var, mlir::Value value) {
    if (symbolTable.count(var))
      return mlir::failure();

    symbolTable.insert(var, value);
    return mlir::success();
  }

  mlir::LogicalResult declareOutput(VarDeclAST *decl) {
    if (availableOutputTable.count(decl->getName()))
      return mlir::failure();

    auto boundsVector = new std::vector<VarBoundsAST>();
    boundsVector->push_back(*decl->getBounds());

    outputTable.insert(decl->getName(), new outputValueList);
    availableOutputTable.insert(decl->getName(), boundsVector);

    return mlir::success();
  }

  mlir::Value mlirGenBitsConstExpr(BitsConstExprAST &expr) {
    if (log2(expr.getValue()) >= expr.getSize()) {
      emitError(loc(expr.loc()), "constant does not fit into specified size");
      return nullptr;
    }

    return builder.create<ConstantNumberOp>(loc(expr.loc()), expr.getValue(), expr.getSize());
  }

  mlir::Value mlirGenBoolConstExpr(BoolConstExprAST &expr) {
    return builder.create<ConstantBooleanOp>(loc(expr.loc()), expr.getValue());
  }

  mlir::Value mlirGenBitsVarExpr(BitsVarExprAST &expr) {
    if (auto variable = symbolTable.lookup(expr.getName())) {
      if (expr.getBounds()->getUpper() == -1 && expr.getBounds()->getLower() == -1) {
        // fix bounds that the parser couldn't infer
        auto newBounds = boundsTable.lookup(expr.getName());
        if (!newBounds) {
          emitError(loc(expr.loc()), "could not find variable bounds");
          return nullptr;
        }

        expr.setBounds(std::unique_ptr<VarBoundsAST>(newBounds));
      }

      return builder.create<SliceOp>(loc(expr.loc()), variable,
                                     expr.getBounds()->getUpper(),
                                     expr.getBounds()->getLower());
    } else {
      emitError(loc(expr.loc()), "unknown variable '")
          << expr.getName() << "'";
      return nullptr;
    }
  }

  mlir::Value mlirGenBitsBinaryExpr(BitsBinaryExprAST &expr) {
    mlir::Value lhs = mlirGenBitsExpr(*expr.getLHS());
    if (!lhs)
      return nullptr;

    mlir::Value rhs = mlirGenBitsExpr(*expr.getRHS());
    if (!rhs)
      return nullptr;

    if (expr.getLHS()->getSize() != expr.getRHS()->getSize()) {
      emitError(loc(expr.loc()), "cannot create expression with different sizes for LHS and RHS ")
          << expr.getLHS()->getSize() << " " << expr.getRHS()->getSize() << "\n";
      return nullptr;
    }

    int size = expr.getLHS()->getSize();

    auto location = loc(expr.loc());
    if (expr.getOp() == "+")
      return builder.create<AddOp>(location, lhs, rhs, size);

    else if (expr.getOp() == "-")
      return builder.create<SubOp>(location, lhs, rhs, size);

    else if (expr.getOp() == "&")
      return builder.create<BitwiseAndOp>(location, lhs, rhs, size);

    else if (expr.getOp() == "|")
      return builder.create<BitwiseOrOp>(location, lhs, rhs, size);

    else if (expr.getOp() == "^")
      return builder.create<BitwiseXorOp>(location, lhs, rhs, size);

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
    auto lhs = expr.getLHS();
    auto rhs = expr.getRHS();

    if (std::find(outputNames.begin(), outputNames.end(), lhs->getName()) == outputNames.end() &&
        std::find(stateNames.begin(), stateNames.end(), lhs->getName()) == stateNames.end()) {
      emitError(loc(lhs->loc()), "no output or state variable named ") << lhs->getName();
      return nullptr;
    }

    mlir::Value rhsValue = mlirGenBitsExpr(*rhs);
    if (!rhsValue)
      return nullptr;

    auto queryBounds = lhs->getBounds();
    if (queryBounds->getUpper() == -1 && queryBounds->getLower() == -1) {
      auto newBounds = boundsTable.lookup(lhs->getName());

      if (!newBounds) {
        emitError(loc(lhs->loc()), "could not find variable bounds");
        return nullptr;
      }

      lhs->setBounds(std::unique_ptr<VarBoundsAST>(newBounds));
      queryBounds = newBounds;
    }

    if (!availableOutputTable.count(lhs->getName()) ||
        !outputTable.count(lhs->getName())) {
      emitError(loc(expr.loc()), "output variable not available in table");
      return nullptr;
    }

    auto availableBounds = availableOutputTable.lookup(lhs->getName());
    bool found = false;

    for (VarBoundsAST bounds : *availableBounds) {
      if (queryBounds->getUpper() <= bounds.getUpper() &&
          queryBounds->getLower() >= bounds.getLower()) {
        if (queryBounds->getUpper() != bounds.getUpper())
          availableBounds->push_back(VarBoundsAST(queryBounds->getUpper() + 1,
                                                  bounds.getUpper()));

        if (queryBounds->getLower() != bounds.getLower())
          availableBounds->push_back(VarBoundsAST(bounds.getLower(),
                                                  queryBounds->getLower() - 1));

        auto it = std::find(availableBounds->begin(), availableBounds->end(), bounds);
        availableBounds->erase(it);

        found = true;
        break;
      }
    }

    if (!found) {
      emitError(loc(expr.loc()), "no output match found for variable ") << lhs->getName();
      return nullptr;
    }

    auto sliced = builder.create<SliceOp>(loc(expr.loc()), rhsValue,
                                          queryBounds->getUpper(),
                                          queryBounds->getLower());
    auto finalValue = std::tuple<VarBoundsAST *, mlir::Value>(queryBounds, sliced);
    outputTable.lookup(lhs->getName())->push_back(finalValue);

    return sliced;
  }

  mlir::Value mlirGenCompExpr(BoolCompExprAST &expr) {
    mlir::Value lhs = mlirGenBitsExpr(*expr.getLHS());
    if (!lhs)
      return nullptr;

    mlir::Value rhs = mlirGenBitsExpr(*expr.getRHS());
    if (!rhs)
      return nullptr;

    if (expr.getLHS()->getSize() != expr.getRHS()->getSize()) {
      emitError(loc(expr.loc()), "cannot compare expressions with different sizes for LHS and RHS");
      return nullptr;
    }

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

  mlir::LogicalResult mlirGenActions(BitsExprASTList *exprList,
                                     mlir::Value condition) {
    auto savedPos = builder.saveInsertionPoint();
    mlir::Block *trueBlock, *falseBlock;

    if (condition) {
      trueBlock = builder.createBlock(condition.getParentRegion(),
                                      condition.getParentRegion()->end());
      falseBlock = builder.createBlock(condition.getParentRegion(),
                                       condition.getParentRegion()->end());

      builder.restoreInsertionPoint(savedPos);
      builder.create<EventCall>(condition.getLoc(), condition, trueBlock, falseBlock);
      builder.setInsertionPointToStart(trueBlock);
    }

    mlir::Value value;

    for (auto &e : *exprList) {
      switch (e->getKind()) {
      case block::ExprAST::Expr_Special:
        value = mlirGenBitsAssignExpr(*llvm::cast<BitsAssignExprAST>(e.get()));
        if (!value)
          return mlir::failure();

        break;

      default:
        return mlir::failure();
      }
    }

    if (condition) {
      builder.create<EventDone>(condition.getLoc(), falseBlock);
      builder.setInsertionPointToStart(falseBlock);
    }

    return mlir::success();
  }

  mlir::LogicalResult mlirGenEvents(EventASTList *events, llvm::StringRef funcName) {
    // parse all the "always" events first - these go at the beginning of the function block
    for (auto &e : *events) {
      if (e->getKind() == "always") {
        if (e->getCondition() != nullptr) {
          emitError(loc(e->loc()), "always event with condition");
          return mlir::failure();
        }

        mlirGenActions(e->getAction(), nullptr);
      }
    }

    // then parse the "when" events
    for (auto &e : *events) {
      if (e->getKind() == "when") {
        mlir::Value cond;

        if (e->getCondition() == nullptr) {
          emitError(loc(e->loc()), "when event without condition");
          return mlir::failure();
        }

        if ((cond = mlirGenBoolExpr(*e->getCondition())) == nullptr)
          return mlir::failure();

        if (failed(mlirGenActions(e->getAction(), cond)))
          return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::FuncOp mlirGenHeader(PrototypeAST *proto, PropertyASTList *properties) {
    std::string blockName = proto->getName();
    auto location = loc(proto->loc());

    PropertyAST *input, *output, *state;
    bool inputFound, outputFound, stateFound;
    inputFound = outputFound = stateFound = false;

    for (auto &p : *properties) {
      if (p->getKind() == "input") {
        if (inputFound) {
          emitError(loc(p->loc()), "duplicate input declaration");
          return nullptr;
        } else {
          inputFound = true;
          input = p.get();
        }
      } else if (p->getKind() == "output") {
        if (outputFound) {
          emitError(loc(p->loc()), "duplicate output declaration");
          return nullptr;
        } else {
          outputFound = true;
          output = p.get();
        }
      } else if (p->getKind() == "state") {
        if (stateFound) {
          emitError(loc(p->loc()), "duplicate state declaration");
          return nullptr;
        } else {
          stateFound = true;
          state = p.get();
        }
      }
    }

    // create the function type
    std::vector<VarDeclAST *> inputVars;
    std::vector<mlir::Type> inputTypes, outputTypes;

    if (inputFound) {
      for (auto &var : *input->getVars()) {
        if (boundsTable.count(var->getName())) {
          emitError(loc(var->loc()), "duplicate variable declaration");
          return nullptr;
        }

        inputNames.push_back(var->getName());
        boundsTable.insert(var->getName(), var->getBounds());

        inputVars.push_back(var.get());
        inputTypes.push_back(mlir::IntegerType::get(abs(var->getBounds()->getUpper() -
                                                        var->getBounds()->getLower()) + 1,
                                                    builder.getContext()));
      }
    }

    if (outputFound) {
      for (auto &var : *output->getVars()) {
        if (boundsTable.count(var->getName())) {
          emitError(loc(var->loc()), "duplicate variable declaration");
          return nullptr;
        }

        outputNames.push_back(var->getName());
        boundsTable.insert(var->getName(), var->getBounds());

        outputTypes.push_back(mlir::IntegerType::get(abs(var->getBounds()->getUpper() -
                                                         var->getBounds()->getLower()) + 1,
                                                     builder.getContext()));

        if (failed(declareOutput(var.get()))) {
          emitError(loc(var->loc()), "duplicate output declaration");
          return nullptr;
        }
      }
    }

    if (stateFound) {
      for (auto &var : *state->getVars()) {
        if (boundsTable.count(var->getName())) {
          emitError(loc(var->loc()), "duplicate variable declaration");
          return nullptr;
        }

        stateNames.push_back(var->getName());
        boundsTable.insert(var->getName(), var->getBounds());

        inputVars.push_back(var.get());
        inputTypes.push_back(mlir::IntegerType::get(abs(var->getBounds()->getUpper() -
                                                        var->getBounds()->getLower()) + 1,
                                                    builder.getContext()));
        outputTypes.push_back(mlir::IntegerType::get(abs(var->getBounds()->getUpper() -
                                                         var->getBounds()->getLower()) + 1,
                                                     builder.getContext()));

        if (failed(declareOutput(var.get()))) {
          emitError(loc(var->loc()), "duplicate output declaration for state");
          return nullptr;
        }
      }
    }

    // create the function itself
    auto funcType = builder.getFunctionType(inputTypes, outputTypes);
    auto function = mlir::FuncOp::create(location, blockName, funcType);
    auto &entryBlock = *function.addEntryBlock();

    for (const auto &nameValue : llvm::zip(inputVars, entryBlock.getArguments())) {
      if (failed(declareInput(std::get<0>(nameValue)->getName(),
                              std::get<1>(nameValue)))) {
        emitError(loc(std::get<0>(nameValue)->loc()),
                  "duplicate input declaration");
        return nullptr;
      }
    }

    builder.setInsertionPointToStart(&entryBlock);
    return function;
  }

  mlir::Value mlirGenMerge(Location blockLoc, llvm::StringRef name) {
    auto varBounds = boundsTable.lookup(name);

    std::vector<VarBoundsAST *> orderedBounds;
    std::vector<mlir::Value> orderedValues;
    int dist;

    for (auto &tuple : *outputTable.lookup(name)) {
      auto newBounds = std::get<0>(tuple);
      auto newValue = std::get<1>(tuple);

      auto pos = std::find_if(orderedBounds.begin(),
                              orderedBounds.end(),
                              [newBounds](VarBoundsAST *val) {
                                return newBounds->getUpper() > val->getLower();
                              });

      dist = std::distance(orderedBounds.begin(), pos);
      orderedBounds.insert(pos, newBounds);
      orderedValues.insert(orderedValues.begin() + dist, newValue);
    }

    if (orderedBounds.empty()) {
      auto newValue = builder.create<ConstantEmptyOp>(loc(blockLoc), "X",
                                                      abs(varBounds->getUpper() -
                                                          varBounds->getLower()) + 1);
      orderedBounds.push_back(varBounds);
      orderedValues.push_back(newValue);
    }

    if (orderedBounds.front()->getUpper() != varBounds->getUpper()) {
      auto newBounds = new VarBoundsAST(orderedBounds.front()->getUpper() + 1, varBounds->getUpper());
      auto newValue = builder.create<ConstantEmptyOp>(loc(blockLoc), "X",
                                                      abs(newBounds->getUpper() -
                                                          newBounds->getLower()) + 1);

      orderedBounds.insert(orderedBounds.begin(), newBounds);
      orderedValues.insert(orderedValues.begin(), newValue);
    }

    if (orderedBounds.back()->getLower() != varBounds->getLower()) {
      auto newBounds = new VarBoundsAST(varBounds->getLower(), orderedBounds.back()->getLower() - 1);
      auto newValue = builder.create<ConstantEmptyOp>(loc(blockLoc), "X",
                                                      abs(newBounds->getUpper() -
                                                          newBounds->getLower()) + 1);

      orderedBounds.push_back(newBounds);
      orderedValues.push_back(newValue);
    }

    // iteratively fix gaps
    while (true) {
      bool done = true;

      for (auto it = orderedBounds.begin(); it != orderedBounds.end(); it++) {
        auto next = std::next(it);
        if (next == orderedBounds.end())
          break;

        if ((*it)->getLower() != (*next)->getUpper() + 1) {
          auto newBounds = new VarBoundsAST((*next)->getUpper() + 1, (*it)->getLower() - 1);
          auto newValue = builder.create<ConstantEmptyOp>(loc(blockLoc),
                                                          "X",
                                                          abs(newBounds->getUpper() -
                                                              newBounds->getLower()) + 1);

          dist = std::distance(orderedBounds.begin(), next);
          orderedBounds.insert(next, newBounds);
          orderedValues.insert(orderedValues.begin() + dist, newValue);

          done = false;
          break;
        }
      }

      if (done)
        break;
    }

    return builder.create<MergeOp>(loc(blockLoc), orderedValues);
  }

  mlir::FuncOp mlirGenBlock(BlockAST &blockAST) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
    llvm::ScopedHashTableScope<llvm::StringRef, VarBoundsAST *> boundsScope(boundsTable);
    llvm::ScopedHashTableScope<llvm::StringRef, outputValueList *> outputScope(outputTable);
    llvm::ScopedHashTableScope<llvm::StringRef, std::vector<VarBoundsAST> *> availableScope(availableOutputTable);

    mlir::FuncOp function(mlirGenHeader(blockAST.getProto(),
                                        blockAST.getProperties()));

    if (failed(mlirGenEvents(blockAST.getEvents(),
                             function.getName()))) {
      function.erase();
      return nullptr;
    }

    std::vector<mlir::Value> returnArgs;

    // finally zip up the outputs
    for (auto name : outputNames) {
      auto value = mlirGenMerge(blockAST.getProto()->loc(), name);
      if (!value)
        return nullptr;

      returnArgs.push_back(value);
    }

    for (auto name : stateNames) {
      auto value = mlirGenMerge(blockAST.getProto()->loc(), name);
      if (!value)
        return nullptr;

      returnArgs.push_back(value);
    }

    builder.create<ReturnOp>(loc(blockAST.getProto()->loc()), (llvm::ArrayRef<mlir::Value>) returnArgs);
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