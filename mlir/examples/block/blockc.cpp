
#include "block/Dialect.h"
#include "block/MLIRGen.h"
#include "block/Parser.h"
#include <memory>

#include "mlir/Analysis/Verifier.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Parser.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace block;
namespace cl = llvm::cl;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input block file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

std::unique_ptr<block::ModuleAST> parseInputFile(llvm::StringRef filename) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(filename);

  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }

  auto buffer = fileOrErr.get()->getBuffer();

  LexerBuffer lexer(buffer.begin(), buffer.end(), std::string(filename));
  Parser parser(lexer);
  return parser.parseModule();
}

int dumpMLIR(ModuleAST *moduleAST)
{
  mlir::registerDialect<mlir::block::BlockDialect>();
  mlir::MLIRContext context;

  mlir::OwningModuleRef module = mlirGen(context, *moduleAST);
  module->dump();
}

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "block compiler\n");

  auto moduleAST = parseInputFile(inputFilename);
  dumpMLIR(moduleAST.get());
}