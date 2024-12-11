#include "pmp_read/read_json/Json_PMP_Parser.hxx"

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

namespace fs = std::filesystem;

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

  auto context = std::make_shared<PMP_Simpleboot_Parsing_Context>();
  Json_PMP_Parser<PMP_Simpleboot_Parsing_Context> parser_simpleboot(
    should_parse_objective, should_parse_normalization, should_parse_matrix,
    [&](PMP_File_Parse_Result &&value) { result = std::move(value); },
    context);

  auto& parser = parser_default;
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
