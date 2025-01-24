#include "mathematica_parse_util.hxx"

std::ostream &operator<<(std::ostream &os, const MMA_ELEMENT &v)
{
  switch(v.index())
    {
    case MMA_ELEMENT_Invalid: os << "[invalid]"; break;
      case MMA_ELEMENT_Expression: {
        const MMA_EXPR &elmt = AS_MMA_ELEMENT(v, Expression);
        switch(elmt.index())
          {
            //case MMA_EXPR_Invalid:
            //	os << "[invalid]";
            //	break;
            case MMA_EXPR_Number: {
              int prev_prec = std::cout.precision();
              os << "[Number " << std::setprecision(15)
                 << AS_MMA_EXPR(elmt, Number) << std::setprecision(prev_prec)
                 << "]";
            }
            break;
          case MMA_EXPR_Polynomial:
            os << "[Polynomial " << AS_MMA_EXPR(elmt, Polynomial) << "]";
            break;
          default: LOGIC_ERROR(DEBUG_STRING(elmt.index()));
          }
        break;
      }
    case MMA_ELEMENT_Operator:
      os << "[Operator " << AS_MMA_ELEMENT(v, Operator) << "]";
      break;
    default: LOGIC_ERROR(DEBUG_STRING(v.index()));
    }
  return os;
}

std::ostream &operator<<(std::ostream &os, const MMA_TOKEN &v)
{
  switch(v.index())
    {
    case MMA_TOKEN_Invalid: os << "[invalid]"; break;
    case MMA_TOKEN_Integer:
      os << "[Integer " << AS_MMA_TOKEN(v, Integer) << "]";
      break;
    case MMA_TOKEN_Real:
      os << "[Float " << AS_MMA_TOKEN(v, Real) << "]";
      break;
    case MMA_TOKEN_Operator:
      os << "[Operator " << AS_MMA_TOKEN(v, Operator) << "]";
      break;
    case MMA_TOKEN_Symbol:
      os << "[Symbol " << AS_MMA_TOKEN(v, Symbol) << "]";
      break;
    case MMA_TOKEN_String:
      os << "[String " << AS_MMA_TOKEN(v, String) << "]";
      break;
    default: break;
    }

  // std::visit([&os](auto &&arg) { os << arg; }, v);
  return os;
}

std::string to_string(const MMA_ELEMENT &v)
{
  std::stringstream ss;
  ss << v;
  return ss.str();
}

std::string to_string(const MMA_TOKEN &v)
{
  std::stringstream ss;
  ss << v;
  return ss.str();
}
