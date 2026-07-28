#include "pmp_read/read_json/Json_PMP_Parser.hxx"
#include "pmp_read/simpleboot/PMP_Simpleboot_Parsing_Context.hxx"
#include "pmp_read/simpleboot/data_provider/Simpleboot_Data_Provider.hxx"

#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include "sdpb_util/json/parse_json.hxx"

namespace fs = std::filesystem;

Simpleboot_Parameters
parse_simpleboot_parameter_file(const std::filesystem::path &param_file);

PMP_File_Parse_Result read_json(
  const std::filesystem::path &input_path, const int64_t max_num_poles,
  const bool should_parse_objective, const bool should_parse_normalization,
  const std::function<bool(size_t matrix_index)> &should_parse_matrix,
  const std::shared_ptr<PMP_Simpleboot_Parsing_Context<>> &simpleboot_context)
{
  PMP_File_Parse_Result result;

  if(simpleboot_context != nullptr)
    {
      Json_PMP_Parser<PMP_Simpleboot_Parsing_Context<>> parser(
        max_num_poles, should_parse_objective, should_parse_normalization,
        should_parse_matrix,
        [&](PMP_File_Parse_Result &&value) { result = std::move(value); },
        simpleboot_context);

      parse_json(input_path, parser);
    }
  else
    {
      Json_PMP_Parser parser(
        max_num_poles, should_parse_objective, should_parse_normalization,
        should_parse_matrix,
        [&](PMP_File_Parse_Result &&value) { result = std::move(value); });

      parse_json(input_path, parser);
    }
  return result;
}
