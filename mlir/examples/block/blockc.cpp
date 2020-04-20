
#include "block/Parser.h"

#include "llvm/Support/CommandLine.h"

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

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "block compiler\n");

  auto moduleAST = parseInputFile(inputFilename);
}