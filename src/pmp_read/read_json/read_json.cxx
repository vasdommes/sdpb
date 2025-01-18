#include "pmp_read/read_json/Json_PMP_Parser.hxx"
#include "pmp_read/simpleboot/PMP_Simpleboot_Parsing_Context.hxx"
#include "pmp_read/simpleboot/data_provider/Simpleboot_Data_Provider.hxx"

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>

namespace fs = std::filesystem;

Simpleboot_Parameters
parse_simpleboot_parameter_file(const std::filesystem::path &param_file);

PMP_File_Parse_Result read_json(
  const std::filesystem::path &input_path, bool should_parse_objective,
  bool should_parse_normalization,
  const std::function<bool(size_t matrix_index)> &should_parse_matrix,
  const std::shared_ptr<PMP_Simpleboot_Parsing_Context<>> &simpleboot_context)
{
  std::ifstream input_file(input_path);
  rapidjson::IStreamWrapper wrapper(input_file);
  PMP_File_Parse_Result result;

  rapidjson::ParseResult res;
  try
    {
      rapidjson::Reader reader;
      if(simpleboot_context != nullptr)
        {
          Json_PMP_Parser<PMP_Simpleboot_Parsing_Context<>> parser(
            should_parse_objective, should_parse_normalization,
            should_parse_matrix,
            [&](PMP_File_Parse_Result &&value) { result = std::move(value); },
            simpleboot_context);

          res = reader.Parse(wrapper, parser);
        }
      else
        {
          Json_PMP_Parser parser(
            should_parse_objective, should_parse_normalization,
            should_parse_matrix,
            [&](PMP_File_Parse_Result &&value) { result = std::move(value); });

          res = reader.Parse(wrapper, parser);
        }
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
