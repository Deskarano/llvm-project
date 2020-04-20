
#ifndef MLIR_BLOCK_PARSER_H
#define MLIR_BLOCK_PARSER_H

#include "block/AST.h"
#include "block/Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

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
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse blocks one at a time and accumulate in this vector.
    std::vector<BlockAST> blocks;
    while (auto f = parseDefinition()) {
      blocks.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof)
        break;
    }

    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(blocks));
  }

private:
  Lexer &lexer;

  std::unique_ptr<VarBoundsAST> parseBounds() {
    if (lexer.getCurToken() != '[')
      return parseError<VarBoundsAST>("[", "to start bounds");

    lexer.consume(Token('['));
    if (lexer.getCurToken() != tok_number)
      return parseError<VarBoundsAST>("number", "for value upper bound");

    int64_t upper = lexer.getValue();
    lexer.consume(tok_number);

    if (lexer.getCurToken() == ':')
    {
      lexer.consume(Token(':'));
      if (lexer.getCurToken() != tok_number)
        return parseError<VarBoundsAST>("number", "for value lower bound");

      int64_t lower = lexer.getValue();
      lexer.consume(tok_number);

      if (lexer.getCurToken() != ']')
        return parseError<VarBoundsAST>("]", "to end bounds");

      lexer.consume(Token(']'));
      return std::make_unique<VarBoundsAST>(lower, upper);
    }
    else if(lexer.getCurToken() == ']')
    {
      lexer.consume(Token(']'));
      return std::make_unique<VarBoundsAST>(upper, upper);
    } else
      return parseError<VarBoundsAST>(": or ]", "to separate or end bounds");
  }

  std::unique_ptr<VarExprAST> parseVarExpr() {
    auto loc = lexer.getLastLocation();
    std::string name(lexer.getId());
    std::unique_ptr<VarBoundsAST> bounds;

    lexer.consume(tok_identifier);
    if (lexer.getCurToken() == '[')
      bounds = parseBounds();
    else
      bounds = std::make_unique<VarBoundsAST>(0, 0);

    return std::make_unique<VarExprAST>(loc, name, std::move(bounds));
  }

  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result =
        std::make_unique<ConstExprAST>(std::move(loc), lexer.getValue());

    lexer.consume(tok_number);
    return std::move(result);
  }

  std::unique_ptr<ExprAST> parseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto v = parseExpression();
    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return v;
  }

  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {

    case tok_identifier:
      return parseVarExpr();

    case tok_number:
      return parseNumberExpr();

    case '(':
      return parseParenExpr();

    default:
      llvm::errs() << "unknown token '" << lexer.getCurToken()
                   << "' when expecting an expression\n";
      return nullptr;
    }
  }

  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    // If this is a binop, find its precedence.
    while (true) {
      int tokPrec = getTokPrecedence();

      // If this is a binop that binds at least as tightly as the current binop,
      // consume it, otherwise we are done.
      if (tokPrec < exprPrec)
        return lhs;

      // Okay, we know this is a binop.
      if(lexer.getCurToken() != tok_arith)
        return parseError<ExprAST>("binary operator", "in RHS of expression");

      std::string binOp = lexer.getOperator();
      lexer.consume(tok_arith);
      auto loc = lexer.getLastLocation();

      // Parse the primary expression after the binary operator.
      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      // If BinOp binds less tightly with rhs than the operator after rhs, let
      // the pending operator take rhs as its lhs.
      int nextPrec = getTokPrecedence();
      if (tokPrec < nextPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      // Merge lhs/RHS.
      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                            std::move(lhs), std::move(rhs));
    }
  }

  std::unique_ptr<ExprAST> parseExpression() {
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;

    return parseBinOpRHS(0, std::move(lhs));
  }

  std::unique_ptr<AssignmentExprAST> parseAssignment() {
    auto loc = lexer.getLastLocation();
    auto lhs = parseVarExpr();
    if(!lhs)
      return nullptr;

    if(lexer.getCurToken() != tok_assign)
      return parseError<AssignmentExprAST>("assignment operator", "in assignment expression");
    lexer.consume(tok_assign);

    auto rhs = parseExpression();
    if(!rhs)
      return nullptr;

    return std::make_unique<AssignmentExprAST>(loc, std::move(lhs), std::move(rhs));
  }

  std::unique_ptr<ConditionExprAST> parseCondition() {
    auto loc = lexer.getLastLocation();
    auto lhs = parseExpression();
    if (!lhs)
      return nullptr;

    if(lexer.getCurToken() != tok_comp)
      return parseError<ConditionExprAST>("comparison operator", "in conditional expression");

    std::string op = lexer.getOperator();
    lexer.consume(tok_comp);

    auto rhs = parseExpression();
    if (!rhs)
      return nullptr;

    return std::make_unique<ConditionExprAST>(loc, op,
                                              std::move(lhs),
                                              std::move(rhs));
  }

  std::unique_ptr<ExprASTList> parseAction() {
    auto exprList = std::make_unique<ExprASTList>();

    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to start event action block");

    lexer.consume(Token('{'));
    while (lexer.getCurToken() != '}') {
      auto expr = parseAssignment();
      if (!expr)
        return nullptr;

      exprList->push_back(std::move(expr));
    }

    lexer.consume(Token('}'));
    return exprList;
  }

  std::unique_ptr<EventASTList> parseEvents() {
    auto eventList = std::make_unique<EventASTList>();

    while (lexer.getCurToken() == tok_event) {
      std::string type = lexer.getEvent();
      auto loc = lexer.getLastLocation();
      std::unique_ptr<ConditionExprAST> condition = nullptr;

      lexer.consume(tok_event);
      if (lexer.getCurToken() == '(') {
        lexer.consume(Token('('));

        condition = parseCondition();
        if (!condition)
          return nullptr;

        if (lexer.getCurToken() != ')')
          return parseError<EventASTList>(")", "to end event condition");
        lexer.consume(Token(')'));
      }

      auto action = parseAction();
      if (!action)
        return nullptr;

      eventList->push_back(std::make_unique<EventAST>(type, loc, std::move(action), std::move(condition)));
    }

    return eventList;
  }

  std::unique_ptr<PropertyASTList> parseProperties() {
    auto propertyList = std::make_unique<PropertyASTList>();

    while (lexer.getCurToken() == tok_property) {
      std::string type = lexer.getProperty();
      auto loc = lexer.getLastLocation();
      auto varList = std::make_unique<VarExprASTList>();

      lexer.consume(tok_property);

      if (lexer.getCurToken() != '(')
        return parseError<PropertyASTList>("(", "to start property list");

      lexer.consume(Token('('));
      while (lexer.getCurToken() != ')') {
        auto var = parseVarExpr();
        if (!var)
          return nullptr;

        varList->push_back(std::move(var));
        if (lexer.getCurToken() == ',')
          lexer.consume(Token(','));
      }

      if (lexer.getCurToken() != ')')
        return parseError<PropertyASTList>(")", "to end property list");

      lexer.consume(Token(')'));
      propertyList->push_back(std::make_unique<PropertyAST>(type, loc, std::move(varList)));
    }

    return propertyList;
  }

  /// prototype ::= block id
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_block)
      return parseError<PrototypeAST>("block", "in prototype");
    lexer.consume(tok_block);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("block name", "in prototype");

    std::string fnName(lexer.getId());
    lexer.consume(tok_identifier);

    return std::make_unique<PrototypeAST>(std::move(loc), fnName);
  }

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype properties events
  std::unique_ptr<BlockAST> parseDefinition() {
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    if (lexer.getCurToken() != '{')
      return parseError<BlockAST>('{', "to begin block");

    lexer.consume(Token('{'));

    auto properties = parseProperties();
    if (!properties)
      return nullptr;

    auto events = parseEvents();
    if (!events)
      return nullptr;

    if (lexer.getCurToken() != '}')
      return parseError<BlockAST>("}", "to end block");

    lexer.consume(Token('}'));
    return std::make_unique<BlockAST>(std::move(proto),
                                      std::move(properties),
                                      std::move(events));
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (lexer.getCurToken() == tok_arith)
      return 20;
    else
      return -1;
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template<typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char) curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }

  template<typename R, typename U = const char *>
  std::unique_ptr<R> logicError(U &&context = "") {
    llvm::errs() << "Logic error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): " << context << "\n";
    return nullptr;
  }
};

} // namespace block

#endif // MLIR_BLOCK_PARSER_H