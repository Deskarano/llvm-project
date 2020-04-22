#include "block/Parser.h"

namespace block {

std::unique_ptr<VarBoundsAST> Parser::parseBounds(bool allowSingle) {
  if (lexer.getCurToken() != '[')
    return parseError<VarBoundsAST>("[", "to start bounds");

  lexer.consume(Token('['));
  if (lexer.getCurToken() != tok_const_num)
    return parseError<VarBoundsAST>("number", "for value upper bound");

  int64_t upper = strtol(lexer.getValue().data(),
                         nullptr, 10);
  lexer.consume(tok_const_num);

  if (lexer.getCurToken() == ':') {
    lexer.consume(Token(':'));
    if (lexer.getCurToken() != tok_const_num)
      return parseError<VarBoundsAST>("number", "for value lower bound");

    int64_t lower = strtol(lexer.getValue().data(),
                           nullptr, 10);
    lexer.consume(tok_const_num);

    if (lexer.getCurToken() != ']')
      return parseError<VarBoundsAST>("]", "to end bounds");

    lexer.consume(Token(']'));
    return std::make_unique<VarBoundsAST>(lower, upper);
  } else if (lexer.getCurToken() == ']') {
    if (allowSingle) {
      lexer.consume(Token(']'));
      return std::make_unique<VarBoundsAST>(upper, upper);
    } else
      return parseError<VarBoundsAST>(":", "to separate bounds (no single allowed)");
  } else
    return parseError<VarBoundsAST>(": or ]", "to separate or end bounds");
}

std::unique_ptr<BitsVarExprAST> Parser::parseBitsVar() {
  auto loc = lexer.getLastLocation();
  std::string name(lexer.getId());
  std::unique_ptr<VarBoundsAST> bounds;

  lexer.consume(tok_identifier);
  if (lexer.getCurToken() == '[') {
    bounds = parseBounds(true);
    if (!bounds)
      return nullptr;
  } else
    bounds = std::make_unique<VarBoundsAST>(0, 0);

  return std::make_unique<BitsVarExprAST>(loc, name, std::move(bounds));
}

std::unique_ptr<BitsConstExprAST> Parser::parseBitsConst() {
  auto loc = lexer.getLastLocation();
  auto result = std::make_unique<BitsConstExprAST>(std::move(loc),
                                                   strtol(lexer.getValue().data(),
                                                          nullptr, 10));
  lexer.consume(tok_const_num);
  return result;
}

std::unique_ptr<BitsExprAST> Parser::parseBitsParen() {
  lexer.getNextToken(); // eat (.

  auto v = parseBitsExpression();
  if (!v)
    return nullptr;

  if (lexer.getCurToken() != ')')
    return parseError<BitsExprAST>(")", "to close expression with parentheses");
  lexer.consume(Token(')'));
  return v;
}

std::unique_ptr<BitsExprAST> Parser::parseBitsBinOpRHS(int exprPrec,
                                                       std::unique_ptr<BitsExprAST> lhs) {
  // If this is a binop, find its precedence.
  while (true) {
    int tokPrec = getTokPrecedence();

    // If this is a binop that binds at least as tightly as the current binop,
    // consume it, otherwise we are done.
    if (tokPrec < exprPrec)
      return lhs;

    // Okay, we know this is a binop.
    if (lexer.getCurToken() != tok_arith)
      return parseError<BitsExprAST>("Bits operator", "in RHS of expression");

    std::string binOp = lexer.getOperator();
    lexer.consume(tok_arith);

    auto loc = lexer.getLastLocation();

    // Parse the primary expression after the binary operator.
    auto rhs = parseBitsPrimary();
    if (!rhs)
      return nullptr;

    // If BinOp binds less tightly with rhs than the operator after rhs, let
    // the pending operator take rhs as its lhs.
    int nextPrec = getTokPrecedence();
    if (tokPrec < nextPrec) {
      rhs = parseBitsBinOpRHS(tokPrec + 1, std::move(rhs));
      if (!rhs)
        return nullptr;
    }

    // Merge lhs/RHS.
    lhs = std::make_unique<BitsBinaryExprAST>(std::move(loc), binOp,
                                              std::move(lhs), std::move(rhs));
  }
}

std::unique_ptr<BitsExprAST> Parser::parseBitsPrimary() {
  switch (lexer.getCurToken()) {

  case tok_identifier:
    return parseBitsVar();

  case tok_const_num:
    return parseBitsConst();

  case '(':
    return parseBitsParen();

  default:
    return parseError<BitsExprAST>("expression",  "when parsing primary of bits expression");
  }
}

std::unique_ptr<BitsExprAST> Parser::parseBitsExpression() {
  auto lhs = parseBitsPrimary();
  if (!lhs)
    return nullptr;

  return parseBitsBinOpRHS(0, std::move(lhs));
}

std::unique_ptr<BitsAssignExprAST> Parser::parseBitsAssignment() {
  auto loc = lexer.getLastLocation();
  auto lhs = parseBitsVar();
  if (!lhs)
    return nullptr;

  if (lexer.getCurToken() != tok_assign)
    return parseError<BitsAssignExprAST>("assignment operator", "in assignment expression");
  lexer.consume(tok_assign);

  auto rhs = parseBitsExpression();
  if (!rhs)
    return nullptr;

  return std::make_unique<BitsAssignExprAST>(loc, std::move(lhs), std::move(rhs));
}
}