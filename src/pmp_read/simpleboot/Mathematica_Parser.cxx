#include "Mathematica_Parser.hxx"

const char *
Mathematica_Parser::parse_element(const char *begin, const char *end,
                                  MMA_ELEMENT &result)
{
  const std::function parse_func
    = [this](const char *begin, const char *end, MMA_ELEMENT &result) {
        return parse_MMA_expr(begin, end, result);
      };
  return reset_and_parse(begin, end, parse_func, result);
}
const char *Mathematica_Parser::parse_token(const char *begin, const char *end,
                                            MMA_TOKEN &result)
{
  const std::function parse_func
    = [this](const char *begin, const char *end, MMA_TOKEN &result) {
        return parse_get_token(begin, end, result);
      };
  return reset_and_parse(begin, end, parse_func, result);
}
const char *
Mathematica_Parser::parse_MMA_function(const std::string &name,
                                       const char *begin, const char *end,
                                       MMA_ELEMENT &result)
{
  MMA_PARSER_ERROR("parse_MMA_function() not implemented");
}
void Mathematica_Parser::parse_MMA_symbol(const std::string &name,
                                          MMA_ELEMENT &result)
{
  MMA_PARSER_ERROR("parse_MMA_symbol() not implemented");
}
void Mathematica_Parser::reset(const char *begin, const char *end)
{
  ptr_MMA_begin = begin;
  ptr_MMA_current = begin;
}
const char *
Mathematica_Parser::skip_space_from_left(const char *begin, const char *end)
{
  const char *rslt = std::find_if_not(
    begin, end, [](const char pc) { return std::isspace(pc); });
  if(*rslt == '\\' && *(rslt + 1) == '\n')
    return skip_space_from_left(rslt + 2, end);
  else
    return rslt;
}
const char *
Mathematica_Parser::find_next_MMA_expr_delimiters(const char *begin,
                                                  const char *end)
{
  const char *pstr = find_delimiters(begin, end, MMA_expr_delimiters);
  if(*pstr == '\n' && pstr > begin && pstr < end
     && *(pstr - 1) == '\\') // special case where a line end with "\\\n"
    return find_next_MMA_expr_delimiters(pstr + 1, end);
  else if(*pstr == '\r' && pstr > begin && pstr < end - 1
          && *(pstr - 1) == '\\'
          && *(pstr + 1)
               == '\n') // special case where a line end with "\\\r\n"
    return find_next_MMA_expr_delimiters(pstr + 2, end);
  else
    return pstr;
}
const char *
Mathematica_Parser::find_end_of_number(const char *p, const char *end)
{
  auto q = find_next_MMA_expr_delimiters(p + 1, end);

  if(*q != '*')
    return q;
  auto q2 = skip_space_from_left(q + 1, end);
  if(*q2 != '^')
    return q;
  q = q2;

  // the text has to be in scientific notation.
  q = skip_space_from_left(q + 1, end);
  if(*q == '+' || *q == '-')
    q = skip_space_from_left(q + 1, end);

  q = find_next_MMA_expr_delimiters(q, end);
  return q;
}
const char *
Mathematica_Parser::parse_get_token(const char *begin, const char *end,
                                    MMA_TOKEN &token)
{
  ptr_MMA_current = begin;
  const char *p = skip_space_from_left(begin, end); //skip_space_l(begin,end);
  if(p == end)
    {
      token = std::monostate{};
      return p;
    } // " 2+(3+1)   "
  // parse basic operators
  const auto ops_ch = std::string("()[]{}+-*/^,");
  if(string_containQ(ops_ch, *p))
    {
      SET_MMA_TOKEN(token, *p, Operator);
      return p + 1;
    }
  // parse numbers
  if(isdigit(*p))
    {
      // this is old code with a bug : it can't handle the following case : 1.`200.*^\\\n-5
      /*
                  auto q = find_next_MMA_expr_delimiters(p + 1, end);

                  if (q + 3 < end && *q == '*' && *(q + 1) == '^')  // 1.`200.*^-50  or 1.`200.*^50
                  {
                          if (*(q + 2) == '+' || *(q + 2) == '-')
                                  q = find_next_MMA_expr_delimiters(q + 3, end);
                          else
                                  q = find_next_MMA_expr_delimiters(q + 2, end);
                  }
                  */

      auto q = find_end_of_number(p, end);

      if(std::find_if_not(p, end, isdigit) == q) // integer
        {
          SET_MMA_TOKEN(token, std::stoi(std::string(p, q)), Integer);
        }
      else
        {
          SET_MMA_TOKEN(token, El::BigFloat(parse_number(p, q)), Real);
        }

      return q;
    }
  // parse symbol
  if(isalpha(*p))
    {
      auto q = find_next_MMA_expr_delimiters(p + 1, end);
      SET_MMA_TOKEN(token, std::string(p, q), Symbol);
      return q;
    }
  // parse string
  if(*p == '\"')
    {
      auto q = std::find(p + 1, end, '\"');
      if(q == end)
        {
          RUNTIME_ERROR("Unrecognizable expression: ", std::string(p, q));
        }
      SET_MMA_TOKEN(token, std::string(p + 1, q), String);
      return q + 1;
    }
  // handle "\\\n" case
  if(*p == '\\' && *(p + 1) == '\n')
    {
      RUNTIME_ERROR("the code shouldn't reach here: ", std::string(p, p + 20));
    }
  RUNTIME_ERROR("Unrecognizable expression: ", std::string(p, p + 10));
}
const char *
Mathematica_Parser::parse_MMA_check_op(const char *begin, const char *end,
                                       MMA_TOKEN &token, char op)
{
  const char *pstr = parse_get_token(begin, end, token);
  if(token.index() != MMA_TOKEN_Operator
     || AS_MMA_TOKEN(token, Operator) != op)
    MMA_PARSER_ERROR("parse_MMA_check_op error : expect ", op, ", but I got ",
                     token, " from text ", std::string(begin, 20));
  return pstr;
}
const char *
Mathematica_Parser::parse_MMA_token_as_int(const char *begin, const char *end,
                                           MMA_TOKEN &token, int &int_num)
{
  const char *pstr = parse_get_token(begin, end, token);
  if(token.index() != MMA_TOKEN_Integer)
    MMA_PARSER_ERROR(
      "parse_MMA_get_token_as_int error : expect integer, but I got ", token,
      " from text ", std::string(begin, 20));
  int_num = AS_MMA_TOKEN(token, Integer);
  return pstr;
}
const char *Mathematica_Parser::parse_MMA_expr_as_number(const char *begin,
                                                         const char *end,
                                                         El::BigFloat &result)
{
  MMA_ELEMENT element;
  const char *pstr = parse_MMA_expr(begin, end, element);

  if(element.index() != MMA_ELEMENT_Expression
     || AS_MMA_ELEMENT(element, Expression).index() != MMA_EXPR_Number)
    MMA_PARSER_ERROR("Expected number at : '", std::string(begin, 10),
                     "' but got ", element);

  result = AS_MMA_ELEMENT_Number(element);
  return pstr;
}
const char *
Mathematica_Parser::parse_MMA_element(const char *begin, const char *end,
                                      MMA_ELEMENT &result)
{
  MMA_TOKEN token;
  const char *pstr = parse_get_token(begin, end, token);

  switch(token.index())
    {
    case MMA_TOKEN_Invalid: result = std::monostate{}; return pstr;

    case MMA_TOKEN_Integer:
      SET_MMA_ELEMENT(result, AS_MMA_TOKEN(token, Integer), Expression);
      return pstr;

    case MMA_TOKEN_Real:
      SET_MMA_ELEMENT(result, AS_MMA_TOKEN(token, Real), Expression);
      return pstr;

      case MMA_TOKEN_Operator: {
        switch(char op = AS_MMA_TOKEN(token, Operator))
          {
          case '+':
          case '-':
          case '*':
          case '/':
          case '^': SET_MMA_ELEMENT(result, op, Operator); return pstr;

            case '(': {
              pstr = parse_MMA_expr(pstr, end, result);

              MMA_TOKEN token_next;
              pstr = parse_get_token(pstr, end, token_next);

              if(token_next.index() != MMA_TOKEN_Operator
                 || AS_MMA_TOKEN(token_next, Operator) != ')')
                MMA_PARSER_ERROR("Expected ')' at '",
                                 std::string(pstr - 1, pstr + 3), "' but got ",
                                 token_next);

              return pstr;
            }

          case ',': // end of expression
          case ')':
          case ']':
          case '}':
            //SET_MMA_ELEMENT(result, op, Operator);
            //return pstr;
            result = std::monostate{};
            --pstr;
            return pstr;

          default:
            MMA_PARSER_ERROR("Unexpected operator : ", op, " before ",
                             std::string(pstr, pstr + 10));
            break;
          }
      }
      break;
    case MMA_TOKEN_Symbol:
      if(*pstr == '[')
        {
          pstr = parse_MMA_function(AS_MMA_TOKEN(token, Symbol), pstr + 1, end,
                                    result);
        }
      else
        {
          parse_MMA_symbol(AS_MMA_TOKEN(token, Symbol), result);
        }
      return pstr;
      break;

    case MMA_TOKEN_String:
    default:
      MMA_PARSER_ERROR("Unexpected token : ", token, " before ",
                       std::string(pstr, pstr + 10));
      break;
    }
  return begin;
}
const char *
Mathematica_Parser::parse_MMA_element_as_expression(const char *begin,
                                                    const char *end,
                                                    MMA_ELEMENT &result)
{
  const char *pstr = parse_MMA_element(begin, end, result);

  if(result.index() != MMA_ELEMENT_Expression)
    MMA_PARSER_ERROR("Expecting expression at : '", std::string(begin, 10),
                     "' but got ", result);
  return pstr;
}
bool Mathematica_Parser::parse_MMA_expr_delimitersQ(const MMA_ELEMENT &elmt)
{
  if(elmt.index() == MMA_ELEMENT_Invalid)
    return true;
  if(elmt.index() != MMA_ELEMENT_Operator)
    return false;
  switch(const char op = AS_MMA_ELEMENT(elmt, Operator))
    {
    case ']':
    case ',': return true;
    default: return false;
    }
}
const char *
Mathematica_Parser::parse_MMA_expr_list(const char *begin, const char *end,
                                        std::list<MMA_ELEMENT> &chain)
{
  MMA_ELEMENT elmt;
  const char *pstr = parse_MMA_element(begin, end, elmt);
  if(elmt.index() == MMA_ELEMENT_Operator)
    {
      switch(char op = AS_MMA_ELEMENT(elmt, Operator))
        {
        case '+':
          pstr = parse_MMA_element_as_expression(pstr, end, elmt);
          chain.push_back(std::move(elmt));
          break;
          case '-': {
            pstr = parse_MMA_element_as_expression(pstr, end, elmt);
            MMA_EXPR &expr = AS_MMA_ELEMENT(elmt, Expression);
            if(expr.index() == MMA_EXPR_Number)
              AS_MMA_ELEMENT_Number(elmt) *= -1;
            else
              AS_MMA_ELEMENT_Polynomial(elmt) *= -1;
            chain.push_back(std::move(elmt));
            break;
          }
        default: MMA_PARSER_ERROR("parse_MMA_expr error 1st op = ", op); break;
        }
    }
  else
    {
      chain.push_back(std::move(elmt));
    }

  while(pstr != end)
    {
      pstr = parse_MMA_element(pstr, end, elmt);

      if(parse_MMA_expr_delimitersQ(elmt))
        break;

      if(chain.back().index() == MMA_ELEMENT_Operator
         && elmt.index() == MMA_ELEMENT_Expression)
        {
          chain.emplace_back(std::move(elmt));
          continue;
        }

      if(chain.back().index() == MMA_ELEMENT_Expression
         && elmt.index() == MMA_ELEMENT_Operator)
        {
          chain.emplace_back(std::move(elmt));
          continue;
        }

      if(chain.back().index() == MMA_ELEMENT_Expression
         && elmt.index() == MMA_ELEMENT_Expression)
        {
          chain.emplace_back('*');
          chain.emplace_back(std::move(elmt));
          continue;
        }

      MMA_PARSER_ERROR("parse_MMA_expr_Times error : illegal element : ", elmt,
                       " after ", chain.back(), " before '",
                       std::string(pstr, 10), "'");
    }

  return pstr;
}
void Mathematica_Parser::parse_MMA_expr_add(MMA_EXPR &e1, MMA_EXPR &e2) const
{
  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Number) += AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Polynomial)
    {
      AS_MMA_EXPR(e1, Polynomial) += AS_MMA_EXPR(e2, Polynomial);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Polynomial) += AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Polynomial)
    {
      AS_MMA_EXPR(e2, Polynomial) += AS_MMA_EXPR(e1, Number);
      SET_MMA_EXPR(e1, AS_MMA_EXPR(e2, Polynomial), Polynomial);
      return;
    }

  MMA_PARSER_ERROR("parse_MMA_expr_add unexpected error : e1.index() = ",
                   e1.index(), " e2.index() = ", e2.index());
}
void Mathematica_Parser::parse_MMA_expr_subtract(MMA_EXPR &e1,
                                                 MMA_EXPR &e2) const
{
  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Number) -= AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Polynomial)
    {
      AS_MMA_EXPR(e1, Polynomial) -= AS_MMA_EXPR(e2, Polynomial);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Polynomial) -= AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Polynomial)
    {
      -AS_MMA_EXPR(e2, Polynomial);
      AS_MMA_EXPR(e2, Polynomial) += AS_MMA_EXPR(e1, Number);
      SET_MMA_EXPR(e1, AS_MMA_EXPR(e2, Polynomial), Polynomial);
      return;
    }

  MMA_PARSER_ERROR("parse_MMA_expr_substract unexpected error : e1.index() = ",
                   e1.index(), " e2.index() = ", e2.index());
}
void Mathematica_Parser::parse_MMA_expr_multiply(MMA_EXPR &e1,
                                                 MMA_EXPR &e2) const
{
  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Number) *= AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Polynomial) *= AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Polynomial)
    {
      AS_MMA_EXPR(e2, Polynomial) *= AS_MMA_EXPR(e1, Number);
      SET_MMA_EXPR(e1, AS_MMA_EXPR(e2, Polynomial), Polynomial);
      return;
    }

  MMA_PARSER_ERROR("parse_MMA_expr_multiply unexpected error : e1.index() = ",
                   e1.index(), " e2.index() = ", e2.index());
}
void Mathematica_Parser::parse_MMA_expr_divide(MMA_EXPR &e1,
                                               const MMA_EXPR &e2) const
{
  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Number) /= AS_MMA_EXPR(e2, Number);
      return;
    }

  if(e1.index() == MMA_EXPR_Polynomial && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Polynomial) /= AS_MMA_EXPR(e2, Number);
      return;
    }

  MMA_PARSER_ERROR("parse_MMA_expr_divide unexpected error : e1.index() = ",
                   e1.index(), " e2.index() = ", e2.index());
}
void Mathematica_Parser::parse_MMA_expr_power(MMA_EXPR &e1,
                                              const MMA_EXPR &e2) const
{
  if(e1.index() == MMA_EXPR_Number && e2.index() == MMA_EXPR_Number)
    {
      AS_MMA_EXPR(e1, Number)
        = to_BigFloat(pow(to_Boost_Float(AS_MMA_EXPR(e1, Number)),
                          to_Boost_Float(AS_MMA_EXPR(e2, Number))));
      return;
    }

  MMA_PARSER_ERROR("parse_MMA_expr_power unexpected error : e1.index() = ",
                   e1.index(), " e2.index() = ", e2.index());
}
void Mathematica_Parser::parse_MMA_expr_single_operate(
  std::list<MMA_ELEMENT> &chain, std::list<MMA_ELEMENT>::iterator it) const
{
  auto it_l = std::prev(it);
  auto it_r = std::next(it);
  switch(AS_MMA_ELEMENT(*it, Operator))
    {
    case '+':
      parse_MMA_expr_add(AS_MMA_ELEMENT(*it_l, Expression),
                         AS_MMA_ELEMENT(*it_r, Expression));
      break;
    case '-':
      parse_MMA_expr_subtract(AS_MMA_ELEMENT(*it_l, Expression),
                              AS_MMA_ELEMENT(*it_r, Expression));
      break;
    case '*':
      parse_MMA_expr_multiply(AS_MMA_ELEMENT(*it_l, Expression),
                              AS_MMA_ELEMENT(*it_r, Expression));
      break;
    case '/':
      parse_MMA_expr_divide(AS_MMA_ELEMENT(*it_l, Expression),
                            AS_MMA_ELEMENT(*it_r, Expression));
      break;
    case '^':
      parse_MMA_expr_power(AS_MMA_ELEMENT(*it_l, Expression),
                           AS_MMA_ELEMENT(*it_r, Expression));
      break;
    default:
      MMA_PARSER_ERROR("parse_MMA_precedence unexpected error : illegal op : ",
                       AS_MMA_ELEMENT(*it, Operator));
      break;
    }

  chain.erase(it);
  chain.erase(it_r);
}
int Mathematica_Parser::parse_MMA_precedence(const char op) const
{
  switch(op)
    {
    case '+':
    case '-': return 3;
    case '*':
    case '/': return 2;
    case '^': return 1;
    default:
      MMA_PARSER_ERROR("parse_MMA_precedence unexpected error : illegal op : ",
                       op);
      break;
    }
}
bool Mathematica_Parser::parse_MMA_precedence_orderedQ(
  const MMA_ELEMENT &op1, const MMA_ELEMENT &op2) const
{
  return parse_MMA_precedence(AS_MMA_ELEMENT(op1, Operator))
         <= parse_MMA_precedence(AS_MMA_ELEMENT(op2, Operator));
}
const char *
Mathematica_Parser::parse_MMA_expr(const char *begin, const char *end,
                                   MMA_ELEMENT &result)
{
  std::list<MMA_ELEMENT> chain;
  const char *pstr = parse_MMA_expr_list(begin, end, chain);

  while(chain.size() > 1)
    {
      for(auto op = std::next(chain.begin()); op != chain.end();
          std::advance(op, 2))
        {
          auto next_op = std::next(op, 2);
          if(next_op == chain.end()
             || parse_MMA_precedence_orderedQ(*op, *next_op))
            {
              parse_MMA_expr_single_operate(chain, op);
              break;
            }
        }
    }

  result = chain.back();
  return pstr;
}
