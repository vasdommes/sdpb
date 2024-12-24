#include "parse_MMA_expr.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <filesystem>
#include <string>

int parse_parameter_find_item(const char *begin, const char *end,
                              const std::string &itemname,
                              const char *&begin_item, const char *&end_item,
                              bool required = true)
{
  const std::string item_head("<" + itemname + ">"),
    item_tail("</" + itemname + ">");

  begin_item = std::search(begin, end, item_head.begin(), item_head.end());
  end_item = std::search(begin, end, item_tail.begin(), item_tail.end());

  if(begin_item == end && required == false)
    return 0;
  begin_item = begin_item + item_head.size();

  if(begin <= begin_item && begin_item < end_item && end_item <= end)
    return 1;

  MMA_PARSER_ERROR("parse_parameter_file error : can't process item ",
                   itemname, " correctly.");
  return 0;
}

void load_block_folder(
  const std::string &block_folder,
  std::map<std::pair<std::string, int>,
           std::vector<std::vector<std::vector<El::BigFloat>>>> &blockF);

const char *parse_parameter_file(const char *begin, const char *end)
{
  using namespace param;

  const char *begin_item;
  const char *end_item;
  MMA_TOKEN token;

  parse_parameter_find_item(begin, end, "kappa", begin_item, end_item);
  parse_MMA_token_as_int(begin_item, end_item, token, kappa);

  // this is only used for interval positivity
  if(parse_parameter_find_item(begin, end, "maxderivs", begin_item, end_item,
                               false))
    parse_MMA_token_as_int(begin_item, end_item, token, maxderivs);
  else
    maxderivs = 0;

  El::BigFloat
    dim_temp; // If I directly using dim, the dim.Precision() is not correct
  parse_parameter_find_item(begin, end, "dim", begin_item, end_item);
  parse_MMA_token_as_float(begin_item, end_item, token, dim_temp);

  dim = std::move(
    dim_temp); // somehow without std::move, the precision is not correct
  nu = (dim - 2) / 2;
  Boost_Float r_crossing_4_temp = (3 - 2 * sqrt(Boost_Float(2))) * 4;
  r_crossing_4
    = r_crossing_4_temp; // the precision is correct without std::move

  parse_parameter_find_item(begin, end, "block", begin_item, end_item);
  parse_MMA_token_as_string(begin_item, end_item, token, block_folder);

  parse_parameter_find_item(begin, end, "input", begin_item, end_item);
  const char *pstr = parse_MMA_check_op(begin_item, end_item, token, '{');
  char op;
  std::string filename;
  while(1)
    {
      pstr = parse_MMA_token_as_string(pstr, end_item, token, filename);

      std::cout << "find input files : " << filename << "\n";

      input_files.push_back(std::move(filename));
      pstr = parse_MMA_get_op(pstr, end_item, token, op);
      if(op == '}')
        break;
      if(op != ',')
        MMA_PARSER_ERROR(
          "parse_parameter_file error : expected ',' , but got ", op,
          " before ", std::string(pstr, 20));
    }

  if(parse_parameter_find_item(begin, end, "variables", begin_item, end_item,
                               false)
     != 0)
    {
      pstr = parse_MMA_check_op(begin_item, end_item, token, '{');
      std::string var_name;
      El::BigFloat var_value;

      while(1)
        {
          pstr = parse_MMA_token_as_string(pstr, end_item, token, var_name);
          pstr = parse_MMA_check_op(pstr, end_item, token, ',');
          pstr = parse_MMA_token_as_float(pstr, end_item, token, var_value);

          var_map.emplace(var_name, var_value);

          pstr = parse_MMA_get_op(pstr, end_item, token, op);
          if(op == '}')
            break;
          if(op != ',')
            MMA_PARSER_ERROR(
              "parse_parameter_file error : expected ',' , but got ", op,
              " before ", std::string(pstr, 20));
        }
    }

  //std::cout << std::setprecision(50) << std::fixed;

  El::Output("parameter file processed");
  El::Output("dim=", dim);
  El::Output("kappa=", kappa);
  El::Output("block=", block_folder);

  El::Output("input={");
  for(auto &file : input_files)
    El::Output("  ", file);
  El::Output("}");

  El::Output("variables={");
  for(const auto &[key, value] : var_map)
    El::Output(key, " = ", value, "");
  El::Output("}");

  //load_block_folder(param::block_folder, blockF);
  generate_blockF_key2index(param::block_folder, blockF_key2index);

  return end;
}

int parse_parameter_file(std::filesystem::path &param_file)
{
  std::ifstream input_stream(param_file);
  if(!input_stream.good())
    {
      throw std::runtime_error("Unable to open parameter file: "
                               + param_file.string());
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
      parse_parameter_file(begin, end);
    }
  catch(std::exception &e)
    {
      RUNTIME_ERROR("Error when parsing parameter file ", param_file, ": ",
                    e.what());
    }
  return 1;
}
