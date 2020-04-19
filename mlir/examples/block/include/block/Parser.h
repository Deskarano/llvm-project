
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
    while (auto f = parseDefinition())
    {
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

  std::unique_ptr<VarBoundsAST> parseBounds()
  {
    if(lexer.getCurToken() != '[')
      return parseError<VarBoundsAST>("[", "to start bounds");

    lexer.consume(Token('['));
    if(lexer.getCurToken() != tok_number)
      return parseError<VarBoundsAST>("number", "for value upper bound");

    int64_t upper = lexer.getValue();
    lexer.consume(tok_number);

    if(lexer.getCurToken() != ':')
      return parseError<VarBoundsAST>(":", "to separate upper and lower bounds");

    lexer.consume(Token(':'));
    if(lexer.getCurToken() != tok_number)
      return parseError<VarBoundsAST>("number", "for value lower bound");

    int64_t lower = lexer.getValue();
    lexer.consume(tok_number);

    if(lexer.getCurToken() != ']')
      return parseError<VarBoundsAST>("]", "to end bounds");

    lexer.consume(Token(']'));
    return std::make_unique<VarBoundsAST>(lower, upper);
  }

  std::unique_ptr<VarAST> parseVar()
  {
    auto loc = lexer.getLastLocation();
    std::string name(lexer.getId());
    std::unique_ptr<VarBoundsAST> bounds;

    lexer.consume(tok_identifier);
    if(lexer.getCurToken() == '[')
      bounds = parseBounds();
    else
      bounds = std::make_unique<VarBoundsAST>(0, 0);

    return std::make_unique<VarAST>(loc, name, std::move(bounds));
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
      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
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

  std::unique_ptr<EventASTList> parseEvents()
  {

  }

  template<typename T>
  std::unique_ptr<T> parseProperty(Token tok)
  {
    auto varList = std::make_unique<VarASTList>();
    auto loc = lexer.getLastLocation();
    lexer.consume(tok);

    if(lexer.getCurToken() != '(')
      return parseError<T>("(", "to start property list");

    lexer.consume(Token('('));
    while(lexer.getCurToken() != ')')
    {
      auto var = parseVar();
      if(!var)
        return nullptr;

      varList->push_back(std::move(var));
      if(lexer.getCurToken() == ',')
        lexer.consume(Token(','));
    }

    if(lexer.getCurToken() != ')')
      return parseError<T>(")", "to end property list");

    lexer.consume(Token(')'));
    return std::make_unique<T>(std::move(loc), std::move(varList));
  }

  std::unique_ptr<PropASTGroup> parseProperties()
  {
    bool foundInput, foundOutput, foundState;
    foundInput = foundOutput = foundState = false;

    std::unique_ptr<InputPropAST> inputProperty;
    std::unique_ptr<OutputPropAST> outputProperty;
    std::unique_ptr<StatePropAST> stateProperty;

    while(!foundInput || !foundOutput || !foundState)
    {
      if(lexer.getCurToken() == tok_prop_input)
      {
        if(foundInput)
          return logicError<PropASTGroup>("duplicate input property");
        else
        {
          inputProperty = parseProperty<InputPropAST>(tok_prop_input);
          if(!inputProperty)
            return nullptr;

          foundInput = true;
        }
      }
      else if(lexer.getCurToken() == tok_prop_output)
      {
        if(foundOutput)
          return logicError<PropASTGroup>("duplicate output property");
        else
        {
          outputProperty = parseProperty<OutputPropAST>(tok_prop_output);
          if(!outputProperty)
            return nullptr;

          foundOutput = true;
        }
      }
      else if(lexer.getCurToken() == tok_prop_state)
      {
        if(foundState)
          return logicError<PropASTGroup>("duplicate state property");
        else
        {
          stateProperty = parseProperty<StatePropAST>(tok_prop_state);
          if(!stateProperty)
            return nullptr;

          foundState = true;
        }
      }
      else break;
    }

    auto loc = lexer.getLastLocation();
    if(!foundInput)
    {
      auto emptyVarList = std::make_unique<VarASTList>();
      inputProperty = std::make_unique<InputPropAST>(std::move(loc),
                                                     std::move(emptyVarList));
    }

    if(!foundOutput)
    {
      auto emptyVarList = std::make_unique<VarASTList>();
      outputProperty = std::make_unique<OutputPropAST>(std::move(loc),
                                                       std::move(emptyVarList));
    }

    if(!foundState)
    {
      auto emptyVarList = std::make_unique<VarASTList>();
      stateProperty = std::make_unique<StatePropAST>(std::move(loc),
                                                     std::move(emptyVarList));
    }

    return std::make_unique<PropASTGroup>(std::move(inputProperty),
                                          std::move(outputProperty),
                                          std::move(stateProperty));
  }

  /// prototype ::= block id
  std::unique_ptr<PrototypeAST> parsePrototype()
  {
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

    if(lexer.getCurToken() != '{')
      return parseError<BlockAST>('{', "to begin block");

    lexer.consume(Token('{'));

    auto properties = parseProperties();
    if(!properties)
      return nullptr;

    auto events = parseEvents();
    if(!events)
      return nullptr;

    if(lexer.getCurToken() != '}')
      return parseError<BlockAST>("}", "to end block");

    lexer.consume(Token('}'));
    return std::make_unique<BlockAST>(std::move(proto),
                                         std::move(properties),
                                         std::move(events));
  }

  /// Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // 1 is lowest precedence.
    switch (static_cast<char>(lexer.getCurToken())) {
    case '-':
      return 20;
    case '+':
      return 20;
    case '*':
      return 40;
    default:
      return -1;
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
                 << lexer.getLastLocation().col << "): expected '" << expected
                 << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }

  template <typename R, typename U = const char *>
  std::unique_ptr<R> logicError(U &&context = "")
  {
    llvm::errs() << "Logic error (" << lexer.getLastLocation().line << ", "
                  << lexer.getLastLocation().col << "): " << context;
    return nullptr;
  }
};

} // namespace block

#endif // MLIR_BLOCK_PARSER_H
