
#ifndef MLIR_BLOCK_PARSER_H
#define MLIR_BLOCK_PARSER_H

#include "block/AST.h"
#include "block/Lexer.h"

#include "llvm/Support/raw_ostream.h"

namespace block {

/// This is a simple recursive parser for the Toy language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule();

private:
  Lexer &lexer;

  /// Boolean
  std::unique_ptr<BoolConstExprAST> parseBoolConst();
  std::unique_ptr<BoolCompExprAST> parseBoolComparison();
  std::unique_ptr<BoolExprAST> parseBoolParen();
  std::unique_ptr<BoolExprAST> parseBoolBinOpRHS(int exprPrec, std::unique_ptr<BoolExprAST> lhs);
  std::unique_ptr<BoolExprAST> parseBoolPrimary();
  std::unique_ptr<BoolExprAST> parseBoolExpression();

  /// Bitwise
  std::unique_ptr<BitsVarExprAST> parseBitsVar();
  std::unique_ptr<BitsConstExprAST> parseBitsConst();
  std::unique_ptr<BitsExprAST> parseBitsParen();
  std::unique_ptr<BitsExprAST> parseBitsBinOpRHS(int exprPrec, std::unique_ptr<BitsExprAST> lhs);
  std::unique_ptr<BitsExprAST> parseBitsPrimary();
  std::unique_ptr<BitsExprAST> parseBitsExpression();
  std::unique_ptr<BitsAssignExprAST> parseBitsAssignment();

  /// Common
  std::unique_ptr<VarBoundsAST> parseBounds(bool allowSingle);
  std::unique_ptr<VarDeclAST> parseVarDecl();

  std::unique_ptr<BitsExprASTList> parseAction();
  std::unique_ptr<EventASTList> parseEvents();
  std::unique_ptr<PropertyASTList> parseProperties();
  std::unique_ptr<PrototypeAST> parsePrototype();
  std::unique_ptr<BlockAST> parseDefinition();

  int getArithTokPrecedence();
  int getBoolTokPrecedence();

  template<typename R, typename T, typename U>
  std::unique_ptr<R> parseError(T &&expected, U &&context) {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char) curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }

  template<typename R, typename U>
  std::unique_ptr<R> logicError(U &&context) {
    llvm::errs()
        << "Logic error (" << lexer.getLastLocation().line << ", "
        << lexer.getLastLocation().col << "): " << context << "\n";
    return nullptr;
  }
};

} // namespace block

#endif // MLIR_BLOCK_PARSER_H