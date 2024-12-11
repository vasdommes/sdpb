#pragma once

#include "Json_Positive_Matrix_With_Prefactor_Parser.hxx"
#include "pmp_read/PMP_File_Parse_Result.hxx"
#include "sdpb_util/json/Abstract_Json_Object_Parser.hxx"
#include "sdpb_util/json/Json_Float_Parser.hxx"
#include "sdpb_util/json/Json_Vector_Parser_With_Skip.hxx"

template <class TFloat> class Json_Simpleboot_Float_Parser;

struct PMP_Default_Parsing_Context
{
  template <class TFloat> using Float_Parser = Json_Float_Parser<TFloat>;
};

struct PMP_Simpleboot_Parsing_Context
{
  template <class TFloat>
  using Float_Parser = Json_Simpleboot_Float_Parser<TFloat>;
};

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
  std::shared_ptr<PMP_Simpleboot_Parsing_Context> context;
};

template <class TContext>
class Json_PMP_Parser final
    : public Abstract_Json_Object_Parser<PMP_File_Parse_Result>
{
  using base_type = Abstract_Json_Object_Parser;
  using value_type = PMP_File_Parse_Result;

private:
  template <class TFloat>
  using Float_Parser = typename TContext::template Float_Parser<TFloat>;
  using BigFloat_Parser = Float_Parser<El::BigFloat>;
  using BigFloat_Vector_Parser = Json_Vector_Parser<BigFloat_Parser>;

  using Json_Positive_Matrix_With_Prefactor_Array_Parser
    = Json_Vector_Parser_With_Skip<
      Json_Positive_Matrix_With_Prefactor_Parser<Float_Parser>>;

  PMP_File_Parse_Result result;

  // Nested parsers
  BigFloat_Vector_Parser objective_parser;
  BigFloat_Vector_Parser normalization_parser;
  Json_Positive_Matrix_With_Prefactor_Array_Parser matrices_parser;

public:
  // In practice,
  // - TArgs &&...args = {}
  //   for Json_PMP_Parser<PMP_Default_Parsing_Context>
  // - TArgs &&...args = const shared_ptr<PMP_Simpleboot_Parsing_Context>&
  //   for Json_PMP_Parser<PMP_Simpleboot_Parsing_Context>
  template <class... TArgs>
  Json_PMP_Parser(
    const bool should_parse_objective, const bool should_parse_normalization,
    const std::function<bool(size_t matrix_index)> &should_parse_matrix,
    const std::function<void(PMP_File_Parse_Result &&result)> &on_parsed,
    TArgs &&...args)
      : base_type(false, on_parsed, [] {}),
        objective_parser(
          !should_parse_objective,
          [this](std::vector<El::BigFloat> &&result) {
            this->result.objective = std::move(result);
          },
          [] {}, std::forward<TArgs>(args)...),
        normalization_parser(
          !should_parse_normalization,
          // accept normalization vector:
          [this](std::vector<El::BigFloat> &&result) {
            this->result.normalization = std::move(result);
          },
          [] {}, std::forward<TArgs>(args)...),
        matrices_parser(
          // don't skip matrices array:
          false,
          // accept matrix
          [this](
            Vector_Parse_Result_With_Skip<Polynomial_Vector_Matrix> &&result) {
            this->result.num_matrices = result.num_elements;
            ASSERT_EQUAL(result.indices.size(), result.parsed_elements.size());
            for(size_t i = 0; i < result.indices.size(); ++i)
              {
                this->result.parsed_matrices.emplace(
                  result.indices.at(i),
                  std::move(result.parsed_elements.at(i)));
              }
          },
          [] {
            LOGIC_ERROR(
              R"(Skipping "PositiveMatrixWithPrefactorArray" not allowed)");
          },
          // Skip some matrices according to their indices:
          [&should_parse_matrix](size_t index) {
            return !should_parse_matrix(index);
          },
          std::forward<TArgs>(args)...)
  {}

  // Abstract_Json_Object_Parser interface implementation:

public:
  Abstract_Json_Reader_Handler &element_parser(const std::string &key) override
  {
    if(key == "objective")
      return objective_parser;
    if(key == "normalization")
      return normalization_parser;
    if(key == "PositiveMatrixWithPrefactorArray")
      return matrices_parser;
    RUNTIME_ERROR("Json_PMP_Parser: Unexpected key=", key);
  }

  void clear_result() override
  {
    result.objective.reset();
    result.normalization.reset();
    result.num_matrices = 0;
    result.parsed_matrices.clear();
  }
  value_type get_result() override { return std::move(result); }
  void reset_element_parsers(bool skip) override
  {
    objective_parser.reset(skip);
    normalization_parser.reset(skip);
    matrices_parser.reset(skip);
  }
};
