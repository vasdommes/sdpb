#include "mathematica_parse_util.hxx"

template <> inline El::BigFloat from_MMA_element(const MMA_ELEMENT &element)
{
  return AS_MMA_ELEMENT_Number(element);
}

template <> inline Boost_Float from_MMA_element(const MMA_ELEMENT &element)
{
  return to_Boost_Float(AS_MMA_ELEMENT_Number(element));
}

template <> inline Polynomial from_MMA_element(const MMA_ELEMENT &element)
{
  return AS_MMA_ELEMENT_Polynomial(element);
}

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
