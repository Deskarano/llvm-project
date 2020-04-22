#include "block/Parser.h"

namespace block {

//std::unique_ptr<BitsVarExprAST> Parser::parseBoolVar() {
//  return nullptr;
//}

std::unique_ptr<BoolConstExprAST> Parser::parseBoolConst() {
  auto loc = lexer.getLastLocation();
  auto result = std::make_unique<BoolConstExprAST>(std::move(loc),
                                                   lexer.getValue().compare("true") == 0);
  lexer.consume(tok_const_bool);
  return result;
}

std::unique_ptr<BoolCompExprAST> Parser::parseBoolComparison() {
  auto loc = lexer.getLastLocation();
  auto lhs = parseBitsExpression();
  if(!lhs)
    return nullptr;

  if(lexer.getCurToken() != tok_comp)
    return parseError<BoolCompExprAST>("comparison operator", "when parsing comparison");

  std::string compOp = lexer.getOperator();
  lexer.consume(tok_comp);

  auto rhs = parseBitsExpression();
  if(!rhs)
    return nullptr;

  return std::make_unique<BoolCompExprAST>(loc, compOp, std::move(lhs), std::move(rhs));
}

std::unique_ptr<BoolExprAST> Parser::parseBoolParen() {
  lexer.getNextToken(); // eat (.

  auto v = parseBoolExpression();
  if (!v)
    return nullptr;

  if (lexer.getCurToken() != ')')
    return parseError<BoolExprAST>(")", "to close expression with parentheses");
  lexer.consume(Token(')'));
  return v;
}

std::unique_ptr<BoolExprAST> Parser::parseBoolBinOpRHS(int exprPrec, std::unique_ptr<BoolExprAST> lhs) {
  while (true) {
    int tokPrec = getTokPrecedence();

    if (tokPrec < exprPrec)
      return lhs;

    if (lexer.getCurToken() != tok_bool)
      return parseError<BoolExprAST>("Boolean operator", "in RHS of expression");

    std::string binOp = lexer.getOperator();
    lexer.consume(tok_bool);

    auto loc = lexer.getLastLocation();
    auto rhs = parseBoolPrimary();
    if (!rhs)
      return nullptr;

    int nextPrec = getTokPrecedence();
    if (tokPrec < nextPrec) {
      rhs = parseBoolBinOpRHS(tokPrec + 1, std::move(rhs));
      if (!rhs)
        return nullptr;
    }

    lhs = std::make_unique<BoolBinaryExprAST>(std::move(loc), binOp,
                                              std::move(lhs), std::move(rhs));
  }
}

std::unique_ptr<BoolExprAST> Parser::parseBoolPrimary() {
  switch (lexer.getCurToken()) {
  case tok_const_bool:
    return parseBoolConst();

  case tok_identifier:
  case tok_const_num:
    return parseBoolComparison();

  case '(':
    return parseBoolParen();

  default:
    return parseError<BoolExprAST>("expression", "when parsing primary of bits expression");
  }
}

std::unique_ptr<BoolExprAST> Parser::parseBoolExpression() {
  auto lhs = parseBoolPrimary();
  if (!lhs)
    return nullptr;

  return parseBoolBinOpRHS(0, std::move(lhs));
}

}