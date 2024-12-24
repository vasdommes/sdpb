#pragma once

#include "PMP_Simpleboot_Parsing_Context.hxx"

struct PMP_Simpleboot_Parsing_Context;

template <class TFloat>
class Json_Simpleboot_Float_Parser final
    : public Json_String_Element_Parser<TFloat>
{
public:
  using value_type = TFloat;

  Json_Simpleboot_Float_Parser(
    bool skip, const std::function<void(value_type &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::shared_ptr<PMP_Simpleboot_Parsing_Context> &context)
      : Json_String_Element_Parser<TFloat>(skip, on_parsed, on_skipped),
        context(context)
  {}

private:
  const std::shared_ptr<PMP_Simpleboot_Parsing_Context> context;

protected:
  TFloat from_string(const std::string &string_value) override
  {
    RUNTIME_ERROR("TODO not implemented: parse Mathematica expression",
                  DEBUG_STRING(string_value));
  }
};
