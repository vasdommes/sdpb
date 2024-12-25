#pragma once

#include "Mathematica_Simpleboot_Expression_Parser.hxx"
#include "sdpb_util/json/Json_String_Element_Parser.hxx"

#include <memory>

// Parses string with Mathematica expression to BigFloat/Boost_Float number.
template <class TFloat, class TContext>
class Json_Simpleboot_Float_Parser final
    : public Json_String_Element_Parser<TFloat>
{
public:
  using value_type = TFloat;

  Json_Simpleboot_Float_Parser(
    bool skip, const std::function<void(value_type &&)> &on_parsed,
    const std::function<void()> &on_skipped, std::shared_ptr<TContext> context)
      : Json_String_Element_Parser<TFloat>(skip, on_parsed, on_skipped),
        context(std::move(context))
  {}

private:
  const std::shared_ptr<TContext> context;

protected:
  TFloat from_string(const std::string &string_value) override
  {
    auto begin = string_value.c_str();
    auto end = begin + string_value.size();
    return from_MMA_element<TFloat>(
      context->expression_parser.parse(begin, end));
  }
};
