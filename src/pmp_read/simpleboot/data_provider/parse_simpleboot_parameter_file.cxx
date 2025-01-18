#include "Simpleboot_Parameters.hxx"
#include "pmp_read/simpleboot/Mathematica_Parser.hxx"
#include "pmp_read/simpleboot/mathematica_parse_util.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <filesystem>
#include <string>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

bool parse_parameter_find_item(const char *begin, const char *end,
                               const std::string &itemname,
                               const char *&begin_item, const char *&end_item,
                               bool required = true)
{
  const std::string item_head("<" + itemname + ">"),
    item_tail("</" + itemname + ">");

  begin_item = std::search(begin, end, item_head.begin(), item_head.end());
  end_item = std::search(begin, end, item_tail.begin(), item_tail.end());

  if(begin_item == end && required == false)
    return false;
  begin_item = begin_item + item_head.size();

  if(begin <= begin_item && begin_item < end_item && end_item <= end)
    return true;

  RUNTIME_ERROR("parse_parameter_file error : can't process item ", itemname,
                " correctly.");
}

template <typename T>
const char *parse_token(const char *begin, const char *end,
                        Mathematica_Parser &parser, T &result)
{
  MMA_TOKEN token;
  const auto parse_end = parser.parse_token(begin, end, token);
  result = std::get<T>(token);
  return parse_end;
}

const char *parse_MMA_check_op(const char *begin, const char *end,
                               Mathematica_Parser &parser, char op)
{
  MMA_TOKEN token;
  const char *pstr = parser.parse_token(begin, end, token);
  if(token.index() != MMA_TOKEN_Operator
     || AS_MMA_TOKEN(token, Operator) != op)
    RUNTIME_ERROR("parse_MMA_check_op error : expect ", op, ", but I got ",
                  token, " from text ", std::string(begin, 20));
  return pstr;
}

const char *parse_MMA_get_op(const char *begin, const char *end,
                             Mathematica_Parser &parser, char &op)
{
  MMA_TOKEN token;
  const char *pstr = parser.parse_token(begin, end, token);
  if(token.index() != MMA_TOKEN_Operator)
    RUNTIME_ERROR("parse_MMA_get_op error : expect Operator, but I got ",
                  token, " from text ", std::string(begin, 20));
  op = AS_MMA_TOKEN(token, Operator);
  return pstr;
}

void load_block_folder(
  const std::string &block_folder,
  std::map<std::pair<std::string, int>,
           std::vector<std::vector<std::vector<El::BigFloat>>>> &blockF);

const char *parse_simpleboot_parameter_file(const char *begin, const char *end,
                                            Simpleboot_Parameters &params)
{
  const char *begin_item;
  const char *end_item;
  MMA_TOKEN token;

  Mathematica_Parser parser;

  parse_parameter_find_item(begin, end, "kappa", begin_item, end_item);

  parse_token<int>(begin_item, end_item, parser, params.kappa);

  // this is only used for interval positivity
  params.maxderivs = 0;
  if(parse_parameter_find_item(begin, end, "maxderivs", begin_item, end_item,
                               false))
    {
      parse_token<int>(begin_item, end_item, parser, params.maxderivs);
    }

  parse_parameter_find_item(begin, end, "dim", begin_item, end_item);
  parse_token<El::BigFloat>(begin_item, end_item, parser, params.dim);

  params.nu = (params.dim - 2) / 2;
  params.r_crossing_4 = Boost_Float((3 - 2 * sqrt(Boost_Float(2))) * 4);

  parse_parameter_find_item(begin, end, "block", begin_item, end_item);
  parser.parse_token(begin_item, end_item, token);
  params.block_folder = AS_MMA_TOKEN(token, String);

  parse_parameter_find_item(begin, end, "input", begin_item, end_item);

  const char *pstr = parse_MMA_check_op(begin_item, end_item, parser, '{');

  char op;
  while(true)
    {
      parser.parse_token(pstr, end, token);
      std::string filename = AS_MMA_TOKEN(token, String);

      El::Output("find input files : ", filename);

      params.input_files.push_back(std::move(filename));
      pstr = parse_MMA_get_op(pstr, end_item, parser, op);
      if(op == '}')
        break;
      if(op != ',')
        RUNTIME_ERROR("parse_parameter_file error : expected ',' , but got ",
                      op, " before ", std::string(pstr, 20));
    }

  if(parse_parameter_find_item(begin, end, "variables", begin_item, end_item,
                               false)
     != 0)
    {
      pstr = parse_MMA_check_op(begin_item, end_item, parser, '{');
      El::BigFloat var_value;

      while(true)
        {
          parser.parse_token(pstr, end_item, token);
          std::string var_name = AS_MMA_TOKEN(token, String);

          pstr = parse_MMA_check_op(pstr, end_item, parser, ',');
          pstr = parse_token<El::BigFloat>(pstr, end_item, parser, var_value);

          params.var_map.emplace(var_name, var_value);

          pstr = parse_MMA_get_op(pstr, end_item, parser, op);
          if(op == '}')
            break;
          if(op != ',')
            RUNTIME_ERROR(
              "parse_parameter_file error : expected ',' , but got ", op,
              " before ", std::string(pstr, 20));
        }
    }

  //std::cout << std::setprecision(50) << std::fixed;

  El::Output("parameter file processed");
  El::Output("dim=", params.dim);
  El::Output("kappa=", params.kappa);
  El::Output("block=", params.block_folder);

  El::Output("input={");
  for(auto &file : params.input_files)
    El::Output("  ", file);
  El::Output("}");

  El::Output("variables={");
  for(const auto &[key, value] : params.var_map)
    El::Output(key, " = ", value, "");
  El::Output("}");

  //load_block_folder(param::block_folder, blockF);
  // TODO:
  // generate_blockF_key2index(params.block_folder, blockF_key2index);

  return end;
}

Simpleboot_Parameters
parse_simpleboot_parameter_file(const std::filesystem::path &param_file)
{
  Simpleboot_Parameters params;

  std::ifstream input_stream(param_file);
  if(!input_stream.good())
    {
      RUNTIME_ERROR("Unable to open parameter file: ", param_file);
    }

  boost::interprocess::file_mapping mapped_file(
    param_file.c_str(), boost::interprocess::read_only);
  boost::interprocess::mapped_region mapped_region(
    mapped_file, boost::interprocess::read_only);

  try
    {
      const char *begin(
        static_cast<const char *>(mapped_region.get_address())),
        *end(begin + mapped_region.get_size());
      parse_simpleboot_parameter_file(begin, end, params);
    }
  catch(std::exception &e)
    {
      RUNTIME_ERROR("Error when parsing parameter file ", param_file, ": ",
                    e.what());
    }
  return params;
}
