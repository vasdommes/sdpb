#pragma once

#include "Json_Positive_Matrix_With_Prefactor_Parser.hxx"
#include "pmp_read/PMP_File_Parse_Result.hxx"
#include "sdpb_util/json/Abstract_Json_Object_Parser.hxx"
#include "sdpb_util/json/Json_Float_Parser.hxx"
#include "sdpb_util/json/Json_Vector_Parser_With_Skip.hxx"

struct PMP_Parsing_Context
{};

template <class TFloat>
class Json_Mathematica_Float_Parser final
    : public Json_String_Element_Parser<TFloat>
{
public:
  using value_type = TFloat;

  Json_Mathematica_Float_Parser(
    bool skip, const std::function<void(value_type &&)> &on_parsed,
    const std::function<void()> &on_skipped,
    const std::shared_ptr<PMP_Parsing_Context> &context)
      : Json_String_Element_Parser<TFloat>(skip, on_parsed, on_skipped),
        context(context)
  {}

private:
  std::shared_ptr<PMP_Parsing_Context> context;
};

class Json_PMP_Parser final
    : public Abstract_Json_Object_Parser<PMP_File_Parse_Result>
{
  using SizeType = rapidjson::SizeType;
  using Ch = rapidjson::UTF8<>::Ch;

private:
  // template <class TFloat> using Float_Parser = Json_Float_Parser<TFloat>;
  template <class TFloat>
  using Float_Parser = Json_Mathematica_Float_Parser<TFloat>;

  using BigFloat_Vector_Parser
    = Json_Vector_Parser<Float_Parser<El::BigFloat>>;

  using Json_Positive_Matrix_With_Prefactor_Array_Parser
    = Json_Vector_Parser_With_Skip<
      Json_Positive_Matrix_With_Prefactor_Parser<Float_Parser>>;

  std::shared_ptr<PMP_Parsing_Context> context;

  PMP_File_Parse_Result result;

  // Nested parsers
  BigFloat_Vector_Parser objective_parser;
  BigFloat_Vector_Parser normalization_parser;
  Json_Positive_Matrix_With_Prefactor_Array_Parser matrices_parser;

public:
  Json_PMP_Parser(
    bool should_parse_objective, bool should_parse_normalization,
    const std::function<bool(size_t matrix_index)> &should_parse_matrix,
    const std::function<void(PMP_File_Parse_Result &&result)> &on_parsed);
  Abstract_Json_Reader_Handler &
  element_parser(const std::string &key) override;

public:
  void clear_result() override;
  value_type get_result() override;
  void reset_element_parsers(bool skip) override;
};
