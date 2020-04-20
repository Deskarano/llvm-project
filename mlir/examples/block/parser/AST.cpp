//===- AST.cpp - Helper for printing out the Toy AST ----------------------===//
//
// Part of the MLIR Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AST dump for the Toy language.
//
//===----------------------------------------------------------------------===//

#include "block/AST.h"

#include "mlir/ADT/TypeSwitch.h"
#include "mlir/Support/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace block;

namespace {

// RAII helper to manage increasing/decreasing the indentation as we traverse
// the AST
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

/// Helper class that implement the AST tree traversal and print the nodes along
/// the way. The only data member is the current indentation level.
class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
//  void dump(const VarType &type);
  void dump(ExprAST *expr);

  void dump(PropertyASTList *list);
  void dump(PrototypeAST *node);
  void dump(BlockAST *node);

  // Actually print spaces matching the current indentation level
  void indent() {
    for (int i = 0; i < curIndent; i++)
      llvm::errs() << "  ";
  }
  int curIndent = 0;
};

} // namespace

/// Return a formatted string for the location of any node
template <typename T> static std::string loc(T *node) {
  const auto &loc = node->loc();
  return (llvm::Twine("@") + *loc.file + ":" + llvm::Twine(loc.line) + ":" +
      llvm::Twine(loc.col))
      .str();
}

// Helper Macro to bump the indentation level and print the leading spaces for
// the current indentations
#define INDENT()                                                               \
  Indent level_(curIndent);                                                    \
  indent();

/// Dispatch to a generic expressions to the appropriate subclass using RTTI
//void ASTDumper::dump(ExprAST *expr) {
//  mlir::TypeSwitch<ExprAST *>(expr)
//      .Case<BinaryExprAST, CallExprAST, LiteralExprAST, NumberExprAST,
//            PrintExprAST, ReturnExprAST, VarDeclExprAST, VariableExprAST>(
//          [&](auto *node) { this->dump(node); })
//      .Default([&](ExprAST *) {
//        // No match, fallback to a generic message
//        INDENT();
//        llvm::errs() << "<unknown Expr, kind " << expr->getKind() << ">\n";
//      });
//}

void ASTDumper::dump(PropertyASTList *list){
  INDENT();
  llvm::errs() << "Properties: \n";
  for (auto &p : *list)
  {
  }
}

/// Print a function prototype, first the function name, and then the list of
/// parameters names.
void ASTDumper::dump(PrototypeAST *node) {
  INDENT();
  llvm::errs() << "Proto '" << node->getName() << "' " << loc(node) << "'\n";
  indent();
}

/// Print a function, first the prototype and then the body.
void ASTDumper::dump(BlockAST *node) {
  INDENT();
  llvm::errs() << "Block \n";
  dump(node->getProto());
  dump(node->getProperties());
//  dump(node->getEvents())
}

/// Print a module, actually loop over the functions and print them in sequence.
void ASTDumper::dump(ModuleAST *node) {
  INDENT();
  llvm::errs() << "Module:\n";
  for (auto &b : *node)
    dump(&b);
}

namespace toy {

// Public API
void dump(ModuleAST &module) { ASTDumper().dump(&module); }

} // namespace block
