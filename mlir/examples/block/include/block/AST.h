#ifndef MLIR_BLOCK_AST_H_
#define MLIR_BLOCK_AST_H_

#include "block/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <vector>

namespace block {

/// Expressions
class ExprAST {
public:
  enum ExprASTType {
    Expr_Bits,
    Expr_Bool,
  };

  enum ExprASTKind {
    Expr_Const,
    Expr_Var,
    Expr_BinOp,
    Expr_Special
  };

  ExprAST(ExprASTKind kind, ExprASTType type, Location location)
      : type(type), kind(kind), location(location) {}

  virtual ~ExprAST() = default;
  ExprASTType getType() const { return type; }
  ExprASTKind getKind() const { return kind; }
  const Location &loc() { return location; }

private:
  const ExprASTType type;
  const ExprASTKind kind;
  Location location;
};

class BitsExprAST : public ExprAST {
public:
  BitsExprAST(ExprASTKind kind, Location location)
      : ExprAST(kind, Expr_Bits, location) {}
};

using BitsExprASTList = std::vector<std::unique_ptr<BitsExprAST>>;

class VarBoundsAST {
public:
  VarBoundsAST(int64_t lower, int64_t upper) :
      lower(lower), upper(upper) {}

  int64_t getLower() { return lower; }
  int64_t getUpper() { return upper; }

private:
  int64_t lower;
  int64_t upper;
};

class BitsVarExprAST : public BitsExprAST {
public:
  BitsVarExprAST(Location location, llvm::StringRef name,
                 std::unique_ptr<VarBoundsAST> bounds) :
      BitsExprAST(Expr_Var, location),
      name(name), bounds(std::move(bounds)) {}

  llvm::StringRef getName() { return name; }
  VarBoundsAST *getBounds() { return bounds.get(); }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bits &&
        c->getKind() == Expr_Var;
  }

private:
  std::string name;
  std::unique_ptr<VarBoundsAST> bounds;
};

using BitsVarExprASTList = std::vector<std::unique_ptr<BitsVarExprAST>>;

class BitsConstExprAST : public BitsExprAST {
public:
  BitsConstExprAST(Location loc, int64_t val) :
      BitsExprAST(Expr_Const, loc), val(val) {}

  int64_t getValue() { return val; }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bits &&
        c->getKind() == Expr_Const;
  }

private:
  int64_t val;
};

class BitsBinaryExprAST : public BitsExprAST {
public:
  BitsBinaryExprAST(Location loc, std::string op,
                    std::unique_ptr<BitsExprAST> lhs,
                    std::unique_ptr<BitsExprAST> rhs)
      : BitsExprAST(Expr_BinOp, loc), op(op),
        lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  llvm::StringRef getOp() { return op; }
  BitsExprAST *getLHS() { return lhs.get(); }
  BitsExprAST *getRHS() { return rhs.get(); }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bits &&
        c->getKind() == Expr_BinOp;
  }

private:
  std::string op;
  std::unique_ptr<BitsExprAST> lhs, rhs;
};

class BitsAssignExprAST : public BitsExprAST {
public:
  BitsAssignExprAST(Location loc,
                    std::unique_ptr<BitsVarExprAST> lhs,
                    std::unique_ptr<BitsExprAST> rhs) :
      BitsExprAST(Expr_Special, loc),
      lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  BitsVarExprAST *getLHS() { return lhs.get(); }
  BitsExprAST *getRHS() { return rhs.get(); }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bits &&
        c->getKind() == Expr_Special;
  }

private:
  std::unique_ptr<BitsVarExprAST> lhs;
  std::unique_ptr<BitsExprAST> rhs;
};

class BoolExprAST : public ExprAST {
public:
  BoolExprAST(ExprASTKind kind, Location location) :
      ExprAST(kind, Expr_Bool, location) {}
};

class BoolConstExprAST : public BoolExprAST {
public:
  BoolConstExprAST(Location location, bool val) :
      BoolExprAST(Expr_Const, location), val(val) {}

  bool getValue() { return val; }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bool &&
        c->getKind() == Expr_Const;
  }

private:
  Location location;
  bool val;
};

class BoolBinaryExprAST : public BoolExprAST {
public:
  BoolBinaryExprAST(Location loc, std::string op,
                    std::unique_ptr<BoolExprAST> lhs,
                    std::unique_ptr<BoolExprAST> rhs) :
      BoolExprAST(Expr_BinOp, loc), op(op),
      lhs(std::move(lhs)),
      rhs(std::move(rhs)) {}

  llvm::StringRef getOp() { return op; }
  BoolExprAST *getLHS() { return lhs.get(); }
  BoolExprAST *getRHS() { return rhs.get(); }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bool &&
        c->getKind() == Expr_BinOp;
  }

private:
  std::string op;
  std::unique_ptr<BoolExprAST> lhs, rhs;
};

class BoolCompExprAST : public BoolExprAST {
public:
  BoolCompExprAST(Location loc, std::string op,
                  std::unique_ptr<BitsExprAST> lhs,
                  std::unique_ptr<BitsExprAST> rhs) :
      BoolExprAST(Expr_BinOp, loc), op(op),
      lhs(std::move(lhs)), rhs(std::move(rhs)) {}

  llvm::StringRef getOp() { return op; }
  BitsExprAST *getLHS() { return lhs.get(); }
  BitsExprAST *getRHS() { return rhs.get(); }

  static bool classof(const ExprAST *c) {
    return c->getType() == Expr_Bool &&
        c->getKind() == Expr_Special;
  }
private:
  std::string op;
  std::unique_ptr<BitsExprAST> lhs, rhs;
};

/// Properties
class VarDeclAST {
public:
  VarDeclAST(Location location, llvm::StringRef name,
             std::unique_ptr<VarBoundsAST> bounds) :
      location(location), name(name), bounds(std::move(bounds)) {}

  virtual ~VarDeclAST() = default;

  const Location &loc() { return location; }
  llvm::StringRef getName() { return name; }
  VarBoundsAST *getBounds() { return bounds.get(); }

private:
  Location location;
  std::string name;
  std::unique_ptr<VarBoundsAST> bounds;
};

using VarDeclASTList = std::vector<std::unique_ptr<VarDeclAST>>;

class PropertyAST {
public:
  PropertyAST(std::string kind,
              Location location,
              std::unique_ptr<VarDeclASTList> vars)
      : kind(kind), location(location), vars(std::move(vars)) {}

  virtual ~PropertyAST() = default;

  llvm::StringRef getKind() const { return kind; }
  const Location &loc() { return location; }
  VarDeclASTList *getVars() { return vars.get(); }

private:
  const std::string kind;
  Location location;
  std::unique_ptr<VarDeclASTList> vars;
};

using PropertyASTList = std::vector<std::unique_ptr<PropertyAST>>;

/// Base class for all event nodes.
class EventAST {
public:
  EventAST(std::string kind,
           Location location,
           std::unique_ptr<BitsExprASTList> action,
           std::unique_ptr<BoolExprAST> condition)
      : kind(kind), location(location),
        action(std::move(action)), condition(std::move(condition)) {}

  virtual ~EventAST() = default;

  llvm::StringRef getKind() const { return kind; }
  const Location &loc() { return location; }
  BitsExprASTList *getAction() { return action.get(); }
  BoolExprAST *getCondition() { return condition.get(); }

private:
  std::string kind;
  Location location;
  std::unique_ptr<BitsExprASTList> action;
  std::unique_ptr<BoolExprAST> condition;
};

using EventASTList = std::vector<std::unique_ptr<EventAST>>;

class PrototypeAST {
  Location location;
  std::string name;

public:
  PrototypeAST(Location location,
               const std::string &name)
      : location(location), name(name) {}

  const Location &loc() { return location; }
  llvm::StringRef getName() const { return name; }
};

class BlockAST {

public:
  BlockAST(std::unique_ptr<PrototypeAST> proto,
           std::unique_ptr<PropertyASTList> properties,
           std::unique_ptr<EventASTList> events)
      : proto(std::move(proto)),
        properties(std::move(properties)),
        events(std::move(events)) {}

  PrototypeAST *getProto() { return proto.get(); }
  PropertyASTList *getProperties() { return properties.get(); }
  EventASTList *getEvents() { return events.get(); }

private:
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<PropertyASTList> properties;
  std::unique_ptr<EventASTList> events;
};

class ModuleAST {
  std::vector<BlockAST> functions;

public:
  ModuleAST(std::vector<BlockAST> functions)
      : functions(std::move(functions)) {}

  auto begin() -> decltype(functions.begin()) { return functions.begin(); }
  auto end() -> decltype(functions.end()) { return functions.end(); }
};

void dump(ModuleAST &);

} // namespace block

#endif // MLIR_BLOCK_AST_H_