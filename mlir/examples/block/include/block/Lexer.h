
#ifndef MLIR_BLOCK_LEXER_H_
#define MLIR_BLOCK_LEXER_H_

#include "llvm/ADT/StringRef.h"

#include <memory>
#include <string>

namespace block {

/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_colon = ':',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',

  tok_eof = -1,

  // commands
  tok_block = -2,
  tok_property = -3,
  tok_event = -4,

  // operators
  tok_arith = -5,
  tok_bool = -6,
  tok_comp = -7,
  tok_assign = -8,

  // primary
  tok_identifier = -9,
  tok_const_num = -10,
  tok_const_bool = -11,
};

/// The Lexer is an abstract base class providing all the facilities that the
/// Parser expects. It goes through the stream one token at a time and keeps
/// track of the location in the file for debugging purposes.
/// It relies on a subclass to provide a `readNextLine()` method. The subclass
/// can proceed by reading the next line from the standard input or from a
/// memory mapped file.
class Lexer {
public:
  /// Create a lexer for the given filename. The filename is kept only for
  /// debugging purposes (attaching a location to a Token).
  Lexer(std::string filename)
      : lastLocation(
      {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}

  virtual ~Lexer() = default;

  /// Look at the current token in the stream.
  Token getCurToken() { return curTok; }

  /// Move to the next token in the stream and return it.
  Token getNextToken() { return curTok = getTok(); }

  /// Move to the next token in the stream, asserting on the current token
  /// matching the expectation.
  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  /// Return the current identifier (prereq: getCurToken() == tok_identifier)
  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  /// Return the current number (prereq: getCurToken() == tok_number)
  llvm::StringRef getValue() {
    assert(curTok == tok_const_num ||
        curTok == tok_const_bool);
    return constantVal;
  }

  /// Return the current property type
  llvm::StringRef getProperty() {
    assert(curTok == tok_property);
    return identifierStr;
  }

  /// Return the current event type
  llvm::StringRef getEvent() {
    assert(curTok == tok_event);
    return identifierStr;
  }

  /// Return the current operator type
  llvm::StringRef getOperator() {
    assert(curTok == tok_arith ||
        curTok == tok_bool ||
        curTok == tok_comp ||
        curTok == tok_assign);
    return operatorStr;
  }

  /// Return the location for the beginning of the current token.
  Location getLastLocation() { return lastLocation; }

  // Return the current line in the file.
  int getLine() { return curLineNum; }

  // Return the current column in the file.
  int getCol() { return curCol; }

private:
  /// Delegate to a derived class fetching the next line. Returns an empty
  /// string to signal end of file (EOF). Lines are expected to always finish
  /// with "\n"
  virtual llvm::StringRef readNextLine() = 0;

  /// Return the next character from the stream. This manages the buffer for the
  /// current line and request the next line buffer to the derived class as
  /// needed.
  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty())
      return EOF;
    ++curCol;
    auto nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty())
      curLineBuffer = readNextLine();
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  ///  Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar))
      lastChar = Token(getNextChar());

    // Save the current location before reading the token characters.
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Identifier: [a-zA-Z][a-zA-Z0-9_]*
    if (isalpha(lastChar)) {
      identifierStr = (char) lastChar;
      while (isalnum((lastChar = Token(getNextChar()))) || lastChar == '_')
        identifierStr += (char) lastChar;

      if (identifierStr == "block")
        return tok_block;

      else if (identifierStr == "input" ||
          identifierStr == "output" ||
          identifierStr == "state")
        return tok_property;

      else if (identifierStr == "always" ||
          identifierStr == "when" ||
          identifierStr == "connect")
        return tok_event;

      else if (identifierStr == "true" ||
          identifierStr == "false") {
        constantVal = identifierStr;
        return tok_const_bool;
      }

      return tok_identifier;
    }

    // Number: [0-9.]+
    if (isdigit(lastChar)) {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar));

      constantVal = numStr;
      return tok_const_num;
    }

    // Operators:
    if (lastChar == '+' || lastChar == '-' || lastChar == '^') {
      operatorStr = lastChar;

      lastChar = Token(getNextChar());
      return tok_arith;
    }

    if (lastChar == '&' || lastChar == '|') {
      operatorStr = lastChar;
      lastChar = Token(getNextChar());

      if (operatorStr[0] == lastChar) {
        operatorStr += lastChar;
        lastChar = Token(getNextChar());
        return tok_bool;
      } else
        return tok_arith;
    }

    if (lastChar == '<' || lastChar == '>' ||
        lastChar == '=' || lastChar == '!') {
      operatorStr = lastChar;
      lastChar = Token(getNextChar());

      if (lastChar == '=') {
        operatorStr += lastChar;
        lastChar = Token(getNextChar());
        return tok_comp;
      } else if (operatorStr[0] == '=')
        return tok_assign;
    }

    if (lastChar == '#') {
      // Comment until end of line.
      do {
        lastChar = Token(getNextChar());
      } while (lastChar != EOF && lastChar != '\n' && lastChar != '\r');

      if (lastChar != EOF)
        return getTok();
    }

    // Check for end of file.  Don't eat the EOF.
    if (lastChar == EOF)
      return tok_eof;

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  /// The last token read from the input.
  Token curTok = tok_eof;

  /// Location for `curTok`.
  Location lastLocation;

  /// If the current Token is an identifier, this string contains the value.
  std::string identifierStr;

  /// If the current Token is a number, this contains the value.
  std::string constantVal;

  /// If the current Token is an operator, this contains the value
  std::string operatorStr;

  /// The last value returned by getNextChar(). We need to keep it around as we
  /// always need to read ahead one character to decide when to end a token and
  /// we can't put it back in the stream after reading from it.
  Token lastChar = Token(' ');

  /// Keep track of the current line number in the input stream
  int curLineNum = 0;

  /// Keep track of the current column number in the input stream
  int curCol = 0;

  /// Buffer supplied by the derived class on calls to `readNextLine()`
  llvm::StringRef curLineBuffer = "\n";
};

/// A lexer implementation operating on a buffer in memory.
class LexerBuffer final : public Lexer {
public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) {}

private:
  /// Provide one line at a time to the Lexer, return an empty string when
  /// reaching the end of the buffer.
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n')
      ++current;
    if (current <= end && *current)
      ++current;
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }
  const char *current, *end;
};
} // namespace block

#endif // MLIR_BLOCK_LEXER_H_