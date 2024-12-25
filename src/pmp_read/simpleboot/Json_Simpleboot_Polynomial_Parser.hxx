#pragma once

#include "Json_Simpleboot_Float_Parser.hxx"
#include "pmp_read/read_json/Json_Polynomial_Parser.hxx"

#include <memory>
#include <string>

// Each polynomial is encoded either as a string (Mathematica expression),
// or as a list of string (each string is a Mathematica expression or number)
template <class TContext>
class Json_Simpleboot_Polynomial_Parser final
    : public Json_Polynomial_Parser<
        Json_Simpleboot_Float_Parser<El::BigFloat, TContext>>
{
  using SizeType = rapidjson::SizeType;
  using Ch = rapidjson::UTF8<>::Ch;

public:
  using value_type = Polynomial;
  using base_type = Json_Polynomial_Parser<
    Json_Simpleboot_Float_Parser<El::BigFloat, TContext>>;

  Json_Simpleboot_Polynomial_Parser(
    const bool skip, const std::function<void(Polynomial &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::shared_ptr<TContext> &context)
      : base_type(skip, on_parsed, on_skipped, context), context(context)
  {}

private:
  const std::shared_ptr<TContext> context;

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
    auto begin = string_value.c_str();
    auto end = begin + string_value.size();
    MMA_ELEMENT element;
    context->expression_parser.parse_element(begin, end, element);
    return from_MMA_element<Polynomial>(element);
  }
};
