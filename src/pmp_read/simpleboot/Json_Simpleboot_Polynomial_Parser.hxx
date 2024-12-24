#pragma once

#include "PMP_Simpleboot_Parsing_Context.hxx"

struct PMP_Simpleboot_Parsing_Context;

class Json_Simpleboot_Polynomial_Parser final
    : public Json_Polynomial_Parser<Json_Simpleboot_Float_Parser<El::BigFloat>>
{
  using SizeType = rapidjson::SizeType;
  using Ch = rapidjson::UTF8<>::Ch;

public:
  using value_type = Polynomial;

  Json_Simpleboot_Polynomial_Parser(
    const bool skip, const std::function<void(Polynomial &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::shared_ptr<PMP_Simpleboot_Parsing_Context> &context)
      : Json_Polynomial_Parser(skip, on_parsed, on_skipped, context),
        context(context)
  {}

private:
  const std::shared_ptr<PMP_Simpleboot_Parsing_Context> context;

public:
  // This code is copied from Json_String_Element_Parser.
  // We don't inherit Json_String_Element_Parser to avoid diamond inheritance problem.
  bool json_string(const Ch *str, SizeType length, bool copy) override
  {
    if(!this->skip)
      {
        std::string string_value(str, length);
        try
          {
            this->result = from_string(string_value);
          }
        catch(std::exception &e)
          {
            RUNTIME_ERROR("'", typeid(*this).name(),
                          "' failed to parse string \"", string_value,
                          "\": ", e.what());
          }
        catch(...)
          {
            RUNTIME_ERROR("'", typeid(*this).name(),
                          "' failed to parse string \"", string_value, "\"");
          }
      }

    this->on_end();
    return true;
  }

  Polynomial from_string(const std::string &string_value)
  {
    RUNTIME_ERROR("TODO not implemented: parse Mathematica expression",
                  DEBUG_STRING(string_value));
  }
};
