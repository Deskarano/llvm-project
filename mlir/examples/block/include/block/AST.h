#ifndef MLIR_BLOCK_AST_H_
#define MLIR_BLOCK_AST_H_

#include "block/Lexer.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"

#include <vector>

namespace block {

class ExprAST {
public:
  enum ExprASTKind {
    Expr_Const,
    Expr_Var,
    Expr_BinOp,
  };

  ExprAST(ExprASTKind kind, Location location)
      : kind(kind), location(location) {}

  virtual ~ExprAST() = default;

  ExprASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

private:
  const ExprASTKind kind;
  Location location;
};

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

class VarExprAST : public ExprAST {
public:
  VarExprAST(Location location, llvm::StringRef name,
             std::unique_ptr<VarBoundsAST> bounds) :
      ExprAST(Expr_Var, location),
      name(name), bounds(std::move(bounds)) {}

  llvm::StringRef getName() { return name; }

  VarBoundsAST *getBounds() { return bounds.get(); }

private:
  std::string name;
  std::unique_ptr<VarBoundsAST> bounds;
};

using VarExprASTList = std::vector<std::unique_ptr<VarExprAST>>;

class ConstExprAST : public ExprAST {
public:
  ConstExprAST(Location loc, int64_t val) :
      ExprAST(Expr_Const, loc), val(val) {}

  int64_t getValue() { return val; }

private:
  int64_t val;
};

class BinaryExprAST : public ExprAST {
public:
  BinaryExprAST(Location loc, char op,
                std::unique_ptr<ExprAST> lhs,
                std::unique_ptr<ExprAST> rhs)
      : ExprAST(Expr_BinOp, loc), op(op),
        lhs(std::move(lhs)),
        rhs(std::move(rhs)) {}

  char getOp() { return op; }

  ExprAST *getLHS() { return lhs.get(); }

  ExprAST *getRHS() { return rhs.get(); }

private:
  char op;
  std::unique_ptr<ExprAST> lhs, rhs;
};

using ExprASTList = std::vector<std::unique_ptr<ExprAST>>;

class ConditionExprAST : public BinaryExprAST {
public:
  ConditionExprAST(Location loc, char op,
                   std::unique_ptr<ExprAST> lhs,
                   std::unique_ptr<ExprAST> rhs) :
      BinaryExprAST(loc, op, std::move(lhs), std::move(rhs)) {}
};

/// Base class for all property nodes.
class PropAST {
public:
  enum PropASTKind {
    Prop_Input, Prop_Output, Prop_State
  };

  PropAST(PropASTKind kind,
          Location location,
          std::unique_ptr<VarExprASTList> vars)
      : kind(kind), location(location), vars(std::move(vars)) {}

  virtual ~PropAST() = default;

  PropASTKind getKind() const { return kind; }

  const Location &loc() { return location; }

  VarExprASTList *getVars() { return vars.get(); }

private:
  const PropASTKind kind;
  Location location;
  std::unique_ptr<VarExprASTList> vars;
};

class InputPropAST : public PropAST {
public:
  InputPropAST(Location location, std::unique_ptr<VarExprASTList> vars) :
      PropAST(Prop_Input, location, std::move(vars)) {}
};

class OutputPropAST : public PropAST {
public:
  OutputPropAST(Location location, std::unique_ptr<VarExprASTList> vars) :
      PropAST(Prop_Output, location, std::move(vars)) {}
};

class StatePropAST : public PropAST {
public:
  StatePropAST(Location location, std::unique_ptr<VarExprASTList> vars) :
      PropAST(Prop_State, location, std::move(vars)) {}
};

class PropASTGroup {
public:
  PropASTGroup(std::unique_ptr<InputPropAST> input,
               std::unique_ptr<OutputPropAST> output,
               std::unique_ptr<StatePropAST> state)
      : input(std::move(input)), output(std::move(output)), state(std::move(state)) {}

private:
  std::unique_ptr<InputPropAST> input;
  std::unique_ptr<OutputPropAST> output;
  std::unique_ptr<StatePropAST> state;

};

/// Base class for all expression nodes.
class EventAST {
public:
  enum EventASTKind {
    Event_When, Event_Always
  };

  EventAST(EventASTKind kind,
           Location location,
           std::unique_ptr<ExprASTList> action)
      : kind(kind), location(location), action(std::move(action)) {}

  virtual ~EventAST() = default;

  EventASTKind getKind() const { return kind; }
  const Location &loc() { return location; }
  ExprASTList *getAction() { return action.get(); }

private:
  const EventASTKind kind;
  Location location;
  std::unique_ptr<ExprASTList> action;
};

class WhenEventAST : public EventAST {
public:
  WhenEventAST(Location location,
               std::unique_ptr<ExprASTList> action,
               std::unique_ptr<ConditionExprAST> condition) :
      EventAST(Event_When, location, std::move(action)), cond(std::move(condition)) {}

  ConditionExprAST *getCondition() { return cond.get(); }

private:
  std::unique_ptr<ConditionExprAST> cond;
};

class AlwaysEventAST : public EventAST {
public:
  AlwaysEventAST(Location location,
                 std::unique_ptr<ExprASTList> action) :
      EventAST(Event_Always, location, std::move(action)) {}
};

/// A block-list of expressions.
using EventASTList = std::vector<std::unique_ptr<EventAST>>;

/// This class represents the "prototype" for a function, which captures its
/// name, and its argument names (thus implicitly the number of arguments the
/// function takes).
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

/// This class represents a function definition itself.
class BlockAST {

public:
  BlockAST(std::unique_ptr<PrototypeAST> proto,
           std::unique_ptr<PropASTGroup> properties,
           std::unique_ptr<EventASTList> events)
      : proto(std::move(proto)),
        properties(std::move(properties)),
        events(std::move(events)) {}

  PrototypeAST *getProto() { return proto.get(); }

  PropASTGroup *getProperties() { return properties.get(); }

  EventASTList *getEvents() { return events.get(); }

private:
  std::unique_ptr<PrototypeAST> proto;
  std::unique_ptr<PropASTGroup> properties;
  std::unique_ptr<EventASTList> events;
};

/// This class represents a list of functions to be processed together
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