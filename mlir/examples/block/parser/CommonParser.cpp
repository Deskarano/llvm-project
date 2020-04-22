#include "block/Parser.h"

namespace block {

std::unique_ptr<VarDeclAST> Parser::parseVarDecl() {
  auto loc = lexer.getLastLocation();
  std::string name(lexer.getId());
  std::unique_ptr<VarBoundsAST> bounds;

  lexer.consume(tok_identifier);
  if (lexer.getCurToken() == '[') {
    bounds = parseBounds(false);
    if (!bounds)
      return nullptr;
  } else
    bounds = std::make_unique<VarBoundsAST>(0, 0);

  return std::make_unique<VarDeclAST>(loc, name, std::move(bounds));
}

std::unique_ptr<BitsExprASTList> Parser::parseAction() {
  auto exprList = std::make_unique<BitsExprASTList>();

  if (lexer.getCurToken() != '{')
    return parseError<BitsExprASTList>("{", "to start event action block");

  lexer.consume(Token('{'));
  while (lexer.getCurToken() != '}') {
    auto expr = parseBitsAssignment();
    if (!expr)
      return nullptr;

    exprList->push_back(std::move(expr));
  }

  lexer.consume(Token('}'));
  return exprList;
}

std::unique_ptr<EventASTList> Parser::parseEvents() {
  auto eventList = std::make_unique<EventASTList>();

  while (lexer.getCurToken() == tok_event) {
    std::string type = lexer.getEvent();
    auto loc = lexer.getLastLocation();
    std::unique_ptr<BoolExprAST> condition = nullptr;

    lexer.consume(tok_event);
    if (lexer.getCurToken() == '(') {
      lexer.consume(Token('('));
      condition = parseBoolExpression();
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

std::unique_ptr<PropertyASTList> Parser::parseProperties() {
  auto propertyList = std::make_unique<PropertyASTList>();

  while (lexer.getCurToken() == tok_property) {
    std::string type = lexer.getProperty();
    auto loc = lexer.getLastLocation();
    auto varList = std::make_unique<VarDeclASTList>();

    lexer.consume(tok_property);

    if (lexer.getCurToken() != '(')
      return parseError<PropertyASTList>("(", "to start property list");

    lexer.consume(Token('('));
    while (lexer.getCurToken() != ')') {
      auto var = parseVarDecl();
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

std::unique_ptr<PrototypeAST> Parser::parsePrototype() {
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

std::unique_ptr<BlockAST> Parser::parseDefinition() {
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

std::unique_ptr<ModuleAST> Parser::parseModule() {
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

int Parser::getTokPrecedence() {
  if (lexer.getCurToken() == tok_arith)
    return 20;
  else if (lexer.getCurToken() == tok_bool)
    return 20;
  else
    return -1;
}

}

