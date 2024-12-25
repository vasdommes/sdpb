#pragma once

#include "Mathematica_Simpleboot_Expression_Parser.hxx"

#include "Json_Simpleboot_Float_Parser.hxx"
#include "Json_Simpleboot_Polynomial_Parser.hxx"
// #include "Simpleboot_Data_Provider.hxx"

template <class TFloat, class TExpressionParser>
class Json_Simpleboot_Float_Parser;
template <class TExpressionParser> class Json_Simpleboot_Polynomial_Parser;

template <class TSimpleboot_Data_Provider>
struct PMP_Simpleboot_Parsing_Context
{
public:
  using Simpleboot_Expression_Parser
    = Mathematica_Simpleboot_Expression_Parser<TSimpleboot_Data_Provider>;
  template <class TFloat>
  using Float_Parser
    = Json_Simpleboot_Float_Parser<TFloat, PMP_Simpleboot_Parsing_Context>;
  using Polynomial_Parser
    = Json_Simpleboot_Polynomial_Parser<PMP_Simpleboot_Parsing_Context>;

public:
  explicit PMP_Simpleboot_Parsing_Context(
    std::shared_ptr<TSimpleboot_Data_Provider> data_provider)
      : expression_parser(std::move(data_provider))
  {}

public:
  Simpleboot_Expression_Parser expression_parser;
};
