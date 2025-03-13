#pragma once

#include "Linear_Combination_Of_Mathematica_Functions.hxx"
#include "mathematica_parse_util.hxx"
#include "sdpb_util/json/Json_String_Element_Parser.hxx"

// Parses string with Mathematica expression to Linear_Combination_Of_Mathematica_Functions.
template <class TContext>
class Json_Simpleboot_Linear_Combination_Of_Functions_Parser final
    : public Json_String_Element_Parser<
        Linear_Combination_Of_Mathematica_Functions>
{
public:
  using value_type = Linear_Combination_Of_Mathematica_Functions;

  Json_Simpleboot_Linear_Combination_Of_Functions_Parser(
    bool skip, const std::function<void(value_type &&)> &on_parsed,
    const std::function<void()> &on_skipped, std::shared_ptr<TContext> context)
      : Json_String_Element_Parser(skip, on_parsed, on_skipped),
        context(std::move(context))
  {}

private:
  const std::shared_ptr<TContext> context;

protected:
  value_type from_string(const std::string &string_value) override
  {
    auto begin = string_value.c_str();
    auto end = begin + string_value.size();
    MMA_ELEMENT element;
    context->linear_combination_of_unevaluated_functions_parser.parse_element(
      begin, end, element);
    return from_MMA_element<Linear_Combination_Of_Mathematica_Functions>(
      element);
  }
};
