#include "Pmp2sdp_Parameters.hxx"

#include "sdpb_util/assert.hxx"

#include <El.hpp>

namespace po = boost::program_options;
namespace fs = std::filesystem;

Pmp2sdp_Parameters::Pmp2sdp_Parameters(int argc, char **argv)
{
  for(int arg(0); arg != argc; ++arg)
    {
      command_arguments.emplace_back(argv[arg]);
    }

  po::options_description options("Basic options");
  options.add_options()("help,h", "Show this helpful message.");
  options.add_options()(
    "input,i", po::value<fs::path>(&input_file),
    "Mathematica, JSON, XML or NSV file with SDP definition");
  // Simpleboot alternative to input_file. TODO rename?
  options.add_options()("simpleboot",
                        po::value<fs::path>(&simpleboot_param_file),
                        "*.m file that defines variables used by simpleboot");
  options.add_options()("output,o",
                        po::value<fs::path>(&output_path)->required(),
                        "Directory to place output");
  options.add_options()(
    "precision,p", po::value<int>(&precision)->required(),
    "The precision, in the number of bits, for numbers in the "
    "computation. ");
  options.add_options()(
    "outputFormat,f",
    po::value<Block_File_Format>(&output_format)
      ->default_value(Block_File_Format::bin),
    "Output format for SDP blocks. Could be either 'bin' or 'json'");
  options.add_options()(
    "zip,z", po::bool_switch(&zip),
    "Store output to zip file instead of plain directory.");
  options.add_options()(
    "verbosity,v",
    po::value<Verbosity>(&verbosity)->default_value(Verbosity::regular),
    "Verbosity.  0 -> no output, 1 -> regular output, 2 -> debug output, 3 -> "
    "trace output");

  // (partial) backward compatibility with pvm2sdp
  po::positional_options_description positional;
  positional.add("precision", 1);
  positional.add("input", 1);
  positional.add("output", 1);

  try
    {
      po::variables_map variables_map;

      po::store(po::command_line_parser(argc, argv)
                  .options(options)
                  .positional(positional)
                  .run(),
                variables_map);

      if(variables_map.count("help") != 0)
        {
          if(El::mpi::Rank() == 0)
            El::Output(options);
          return;
        }

      po::notify(variables_map);

      ASSERT(variables_map.count("simpleboot") + variables_map.count("input")
               == 1,
             "Either --input or --simpleboot option should be specified");

      if(variables_map.count("input") > 0)
        {
          ASSERT(fs::exists(input_file),
                 "Input file does not exist: ", input_file);
          ASSERT(!fs::is_directory(input_file) && input_file != ".",
                 "Input file is a directory, not a file:", input_file);
        }
      if(variables_map.count("simpleboot") > 0)
        {
          ASSERT(fs::exists(simpleboot_param_file),
                 "Simpleboot parameters file does not exist: ",
                 simpleboot_param_file);
          ASSERT(!fs::is_directory(simpleboot_param_file)
                   && simpleboot_param_file != ".",
                 "Simpleboot parameters file is a directory, not a file:",
                 simpleboot_param_file);
        }
    }
  catch(po::error &e)
    {
      El::ReportException(e);
      El::mpi::Abort(El::mpi::COMM_WORLD, 1);
    }
}
bool Pmp2sdp_Parameters::is_valid() const
{
  return !input_file.empty();
}
boost::property_tree::ptree to_property_tree(const Pmp2sdp_Parameters &p)
{
  boost::property_tree::ptree result;

  result.put("input", p.input_file.string());
  result.put("parameter", p.simpleboot_param_file.string());
  result.put("output", p.output_path.string());
  result.put("precision", p.precision);
  result.put("outputFormat", p.output_format);
  result.put("zip", p.zip);
  result.put("verbosity", static_cast<int>(p.verbosity));

  return result;
}