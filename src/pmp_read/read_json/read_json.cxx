#include "pmp_read/read_json/Json_PMP_Parser.hxx"
#include "pmp_read/simpleboot/PMP_Simpleboot_Parsing_Context.hxx"
#include "pmp_read/simpleboot/Simpleboot_Data_Provider.hxx"

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

namespace fs = std::filesystem;

Simpleboot_Parameters
parse_simpleboot_parameter_file(const std::filesystem::path &param_file);

PMP_File_Parse_Result
read_json(const std::filesystem::path &input_path, bool should_parse_objective,
          bool should_parse_normalization,
          const std::function<bool(size_t matrix_index)> &should_parse_matrix)
{
  std::ifstream input_file(input_path);
  rapidjson::IStreamWrapper wrapper(input_file);
  PMP_File_Parse_Result result;

  // TODO here we create two parsers to test compilation

  Json_PMP_Parser<PMP_Default_Parsing_Context> parser_default(
    should_parse_objective, should_parse_normalization, should_parse_matrix,
    [&](PMP_File_Parse_Result &&value) { result = std::move(value); });

  // TODO initialize from command-line input
  const std::filesystem::path params_file;
  const auto params = parse_simpleboot_parameter_file(params_file);
  const auto simpleboot_provider
    = std::make_shared<Simpleboot_Data_Provider>(params);

  const auto context
    = std::make_shared<PMP_Simpleboot_Parsing_Context<Simpleboot_Data_Provider>>(
      simpleboot_provider);
  Json_PMP_Parser<PMP_Simpleboot_Parsing_Context<Simpleboot_Data_Provider>>
    parser_simpleboot(
      should_parse_objective, should_parse_normalization, should_parse_matrix,
      [&](PMP_File_Parse_Result &&value) { result = std::move(value); },
      context);

  auto &parser = parser_default;
  // auto& parser = parser_simpleboot;

  rapidjson::ParseResult res;
  try
    {
      rapidjson::Reader reader;
      res = reader.Parse(wrapper, parser);
    }
  catch(std::exception &e)
    {
      RUNTIME_ERROR("Failed to parse ", input_path,
                    ": offset=", wrapper.Tell(), ": ", e.what());
    }
  if(res.IsError())
    {
      RUNTIME_ERROR("Failed to parse ", input_path, ": offset=", res.Offset(),
                    ": error: ", rapidjson::GetParseError_En(res.Code()));
    }
  return result;
}
