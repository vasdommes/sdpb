#pragma once

#include "sdpb_util/json/Json_Float_Parser.hxx"
#include "sdpb_util/json/Json_Vector_Parser.hxx"
#include "pmp/Polynomial.hxx"

template <class TJson_BigFloat_Parser>
class Json_Polynomial_Parser final
    : public Abstract_Json_Vector_Parser<Polynomial, TJson_BigFloat_Parser>
{
  Polynomial result{0, 0};

public:
  using element_type = El::BigFloat;
  using value_type = Polynomial;

  template <class... TArgs>
  Json_Polynomial_Parser(
    const bool skip, const std::function<void(Polynomial &&)> &on_parsed,
    const std::function<void()> &on_skipped = [] {},
    TArgs &&...bigfloat_parser_args)
      : Abstract_Json_Vector_Parser<Polynomial, TJson_BigFloat_Parser>(
          skip, on_parsed, on_skipped,
          std::forward<TArgs>(bigfloat_parser_args)...)
  {}

  void clear_result() override { result.coefficients.clear(); }
  void on_element_parsed(element_type &&value, size_t index) override
  {
    ASSERT_EQUAL(index, result.coefficients.size());
    result.coefficients.push_back(std::forward<element_type>(value));
  }
  void on_element_skipped(size_t /*index*/) override {}
  value_type get_result() override { return std::move(result); }
};
