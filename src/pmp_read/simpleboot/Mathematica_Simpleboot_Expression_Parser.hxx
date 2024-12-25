#pragma once

#include "parse_MMA_expr.hxx"
#include "sdpb_util/assert.hxx"

#include <memory>
#include <typeinfo>

using MMA_EXPR = std::variant<El::BigFloat, Polynomial>;
#define MMA_EXPR_Number 0
#define MMA_EXPR_Polynomial 1
#define AS_MMA_EXPR(var, T) std::get<MMA_EXPR_##T>(var)
#define SET_MMA_EXPR(var, value, T) var.emplace<MMA_EXPR_##T>(value)

using MMA_ELEMENT = std::variant<std::monostate, MMA_EXPR, char>;
#define MMA_ELEMENT_Invalid 0
#define MMA_ELEMENT_Expression 1
#define MMA_ELEMENT_Operator 2
#define AS_MMA_ELEMENT(var, T) std::get<MMA_ELEMENT_##T>(var)
#define AS_MMA_ELEMENT_Number(var)                                            \
  std::get<MMA_EXPR_Number>(std::get<MMA_ELEMENT_Expression>(var))
#define AS_MMA_ELEMENT_Polynomial(var)                                        \
  std::get<MMA_EXPR_Polynomial>(std::get<MMA_ELEMENT_Expression>(var))
#define SET_MMA_ELEMENT(var, value, T) var.emplace<MMA_ELEMENT_##T>(value)

// Helper functions used by Json_Simpleboot_Float_Parser and Json_Simpleboot_Polynomial_Parser.
// We had to move from_MMA_element() outside of class
// because C++ does not allow partial specialization for member functions

template <class TResult> TResult from_MMA_element(const MMA_ELEMENT &element)
{
  RUNTIME_ERROR("from_MMA_element<TResult>() "
                "not implemented for type TResult=",
                typeid(TResult).name());
}

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

inline std::ostream &operator<<(std::ostream &os, const MMA_ELEMENT &v)
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

template <class TSimpleboot_Data_Provider>
class Mathematica_Simpleboot_Expression_Parser final
{
private:
  const char *ptr_MMA_begin = nullptr;
  const char *ptr_MMA_current = nullptr;
  const char *ptr_MMA_end = nullptr;
  const std::shared_ptr<TSimpleboot_Data_Provider> data_provider;

public:
  explicit Mathematica_Simpleboot_Expression_Parser(
    std::shared_ptr<TSimpleboot_Data_Provider> data_provider)
      : data_provider(std::move(data_provider))
  {}

private:
  void reset(const char *begin, const char *end)
  {
    ptr_MMA_begin = begin;
    ptr_MMA_current = begin;
    ptr_MMA_end = end;
  }

public:
  MMA_ELEMENT parse(const char *begin, const char *end)
  {
    reset(begin, end);

    ASSERT(ptr_MMA_begin != nullptr);
    ASSERT(ptr_MMA_end != nullptr);
    ASSERT(ptr_MMA_begin != ptr_MMA_end);

    MMA_ELEMENT result;
    parse_MMA_expr(ptr_MMA_begin, ptr_MMA_end, result);
    ptr_MMA_current = skip_space_from_left(ptr_MMA_current, ptr_MMA_end);
    ASSERT_EQUAL(ptr_MMA_current, ptr_MMA_end,
                 "Failed to parse Mathematica expression: the last ",
                 ptr_MMA_end - ptr_MMA_current,
                 " characters left unprocessed");

    reset(nullptr, nullptr);
    return result;
  }
};
