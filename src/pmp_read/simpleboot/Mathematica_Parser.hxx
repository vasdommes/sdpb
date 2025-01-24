#pragma once

#include "mathematica_parse_util.hxx"
#include "pmp_read/read_mathematica/parse_SDP/parse_number.hxx"
#include "sdpb_util/assert.hxx"

#include <functional>
#include <list>

class Mathematica_Parser
{
private:
  const char *ptr_MMA_current = nullptr;

public:
  virtual ~Mathematica_Parser() = default;

  const char *
  parse_element(const char *begin, const char *end, MMA_ELEMENT &result);

  const char *
  parse_token(const char *begin, const char *end, MMA_TOKEN &result);

private:
  template <class... TArgs>
  [[noreturn]] void
  parse_error_with_position_and_next_chars(const char *begin, const char *end,
                                           const TArgs &...args)
  {
    const auto pos = ptr_MMA_current - begin;
    constexpr int max_length = 32;
    const auto next_chars = short_string(ptr_MMA_current, end, max_length);
    RUNTIME_ERROR("Failed to parse Mathematica input at position: ", pos,
                  ", next characters: ", next_chars, args...);
  }

  template <class T>
  const char *reset_and_parse(
    const char *begin, const char *end,
    std::function<const char *(const char *, const char *, T &)> parse_func,
    T &result)
  {
    ASSERT(begin != nullptr);
    ASSERT(end != nullptr);
    ASSERT(begin <= end);
    reset(begin, end);
    try
      {
        return parse_func(begin, end, result);
      }
    catch(std::exception &e)
      {
        parse_error_with_position_and_next_chars(begin, end,
                                                 "\nException: ", e.what());
      }
    catch(...)
      {
        parse_error_with_position_and_next_chars(begin, end, "\nException: ");
      }
  }

protected:
  // Virtual functions, to be implemented by inheritors

  // TODO write some generic function parsing logic here,
  // and delegate concrete function evaluation to inheritors
  virtual const char *
  parse_MMA_function(const std::string &name, const char *begin,
                     const char *end, MMA_ELEMENT &result);

  virtual void parse_MMA_symbol(const std::string &name, MMA_ELEMENT &result);

protected:
  void reset(const char *begin, const char *end);

  // Helper function: return up to max_length characters from [begin,end)
  static std::string
  short_string(const char *begin, const char *end, size_t max_length = 16);

  // Parser implementation

  const std::string MMA_expr_delimiters = "()[]{}+-*/^, \t\n\v\f\r";

  static const char *skip_space_from_left(const char *begin, const char *end);

  template <class InputIterator>
  InputIterator find_delimiters(InputIterator begin, InputIterator end,
                                const std::string delimiters);

  const char *
  find_next_MMA_expr_delimiters(const char *begin, const char *end);

  // example : 1.`200.*^-50  or 1.`200.*^50
  // p is a digit
  const char *find_end_of_number(const char *p, const char *end);

  template <typename T>
  static bool string_containQ(const std::string &str, const T substr);

  const char *
  parse_get_token(const char *begin, const char *end, MMA_TOKEN &token);

  const char *parse_MMA_check_op(const char *begin, const char *end,
                                 MMA_TOKEN &token, char op);

  const char *parse_MMA_token_as_int(const char *begin, const char *end,
                                     MMA_TOKEN &token, int &int_num);

  const char *parse_MMA_expr_as_number(const char *begin, const char *end,
                                       El::BigFloat &result);

  const char *
  parse_MMA_element(const char *begin, const char *end, MMA_ELEMENT &result);

  const char *
  parse_MMA_element_as_expression(const char *begin, const char *end,
                                  MMA_ELEMENT &result);

  static bool parse_MMA_expr_delimitersQ(const MMA_ELEMENT &elmt);
  ;

  const char *parse_MMA_expr_list(const char *begin, const char *end,
                                  std::list<MMA_ELEMENT> &chain);
  static void parse_MMA_expr_add(MMA_EXPR &e1, MMA_EXPR &e2);

  static void parse_MMA_expr_subtract(MMA_EXPR &e1, MMA_EXPR &e2);
  static void parse_MMA_expr_multiply(MMA_EXPR &e1, MMA_EXPR &e2);

  static void parse_MMA_expr_divide(MMA_EXPR &e1, const MMA_EXPR &e2);

  static void parse_MMA_expr_power(MMA_EXPR &e1, const MMA_EXPR &e2);

  void
  parse_MMA_expr_single_operate(std::list<MMA_ELEMENT> &chain,
                                std::list<MMA_ELEMENT>::iterator it) const;

  int parse_MMA_precedence(const char op) const;

  bool parse_MMA_precedence_orderedQ(const MMA_ELEMENT &op1,
                                     const MMA_ELEMENT &op2) const;

  const char *
  parse_MMA_expr(const char *begin, const char *end, MMA_ELEMENT &result);
};

template <class InputIterator>
InputIterator
Mathematica_Parser::find_delimiters(InputIterator begin, InputIterator end,
                                    const std::string delimiters)
{
  return std::find_first_of(begin, end, delimiters.begin(), delimiters.end());
}

template <typename T>
bool Mathematica_Parser::string_containQ(const std::string &str,
                                         const T substr)
{
  return str.find(substr) != std::string::npos;
}
