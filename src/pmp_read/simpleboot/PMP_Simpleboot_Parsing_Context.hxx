#pragma once

#include "Mathematica_Simpleboot_Expression_Parser.hxx"

#include "Json_Simpleboot_Float_Parser.hxx"
#include "Json_Simpleboot_Linear_Combination_Of_Functions_Parser.hxx"
#include "Json_Simpleboot_Polynomial_Parser.hxx"
#include "data_provider/Simpleboot_Data_Provider.hxx"

template <class TFloat, class TExpressionParser>
class Json_Simpleboot_Float_Parser;
template <class TExpressionParser> class Json_Simpleboot_Polynomial_Parser;

template <class TSimpleboot_Data_Provider = Simpleboot_Data_Provider>
struct PMP_Simpleboot_Parsing_Context
{
public:
  using Simpleboot_Expression_Parser
    = Mathematica_Simpleboot_Expression_Parser<TSimpleboot_Data_Provider,
                                               /*evaluate_F0_F_FS=*/true>;
  using Simpleboot_Linear_Combination_Of_Functions_Parser
    = Mathematica_Simpleboot_Expression_Parser<TSimpleboot_Data_Provider,
                                               /*evaluate_F0_F_FS=*/false>;
  template <class TFloat>
  using Float_Parser
    = Json_Simpleboot_Float_Parser<TFloat, PMP_Simpleboot_Parsing_Context>;
  using Polynomial_Parser
    = Json_Simpleboot_Polynomial_Parser<PMP_Simpleboot_Parsing_Context>;
  using Objective_Element_Parser
    = Json_Simpleboot_Linear_Combination_Of_Functions_Parser<
      PMP_Simpleboot_Parsing_Context>;
  using Normalization_Element_Parser
    = Json_Simpleboot_Linear_Combination_Of_Functions_Parser<
      PMP_Simpleboot_Parsing_Context>;

public:
  explicit PMP_Simpleboot_Parsing_Context(
    const std::shared_ptr<TSimpleboot_Data_Provider> &data_provider)
      : data_provider(data_provider),
        expression_parser(data_provider),
        linear_combination_of_unevaluated_functions_parser(data_provider)
  {}

public:
  const std::shared_ptr<TSimpleboot_Data_Provider> data_provider;
  Simpleboot_Expression_Parser expression_parser;
  Simpleboot_Linear_Combination_Of_Functions_Parser
    linear_combination_of_unevaluated_functions_parser;
};
