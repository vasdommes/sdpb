#include "Output_SDP/Output_SDP.hxx"
#include "Pmp2sdp_Parameters/Pmp2sdp_Parameters.hxx"
#include "Dual_Constraint_Group.hxx"
#include "write_sdp.hxx"
#include "pmp_read/pmp_read.hxx"
#include "pmp_read/simpleboot/data_provider/Simpleboot_Parameters.hxx"
#include "sdpb_util/Verbosity.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <string>
#include <boost/program_options.hpp>
#include <filesystem>

namespace fs = std::filesystem;
namespace po = boost::program_options;

Simpleboot_Parameters
parse_simpleboot_parameter_file(const std::filesystem::path &param_file);

int main(int argc, char **argv)
{
  Environment env(argc, argv);

  try
    {
      Pmp2sdp_Parameters parameters(argc, argv);
      if(!parameters.is_valid())
        {
          return 0;
        }
      // Print command line
      if(parameters.verbosity >= Verbosity::debug && El::mpi::Rank() == 0)
        {
          std::vector<std::string> arg_list(argv, argv + argc);
          for(const auto &arg : arg_list)
            std::cout << arg << " ";
          std::cout << std::endl;
        }

      Environment::set_precision(parameters.precision);

      Timers timers(env, parameters.verbosity);
      Scoped_Timer timer(timers, "pmp2sdp");

      std::optional<Simpleboot_Parameters> simpleboot_parameters;
      std::vector<fs::path> input_files;
      if(parameters.simpleboot_param_file.empty())
        {
          input_files = {parameters.input_file};
        }
      else
        {
          simpleboot_parameters = parse_simpleboot_parameter_file(
            parameters.simpleboot_param_file);
          input_files = simpleboot_parameters->input_files;
          if(!parameters.input_file.empty() && El::mpi::Rank() == 0)
            {
              PRINT_WARNING("--input=", parameters.input_file,
                            " will be ignored, list of input files is read "
                            "from parameter file --simpleboot=",
                            parameters.simpleboot_param_file);
            }
        }

      auto pmp = read_polynomial_matrix_program(
        env, input_files, parameters.max_num_poles, parameters.verbosity, timers, simpleboot_parameters);

      Output_SDP sdp(pmp, parameters.command_arguments, timers);
      write_sdp(parameters.output_path, sdp, pmp, parameters.output_format,
                parameters.zip, timers, parameters.verbosity);
      if(parameters.verbosity >= Verbosity::regular && El::mpi::Rank() == 0)
        {
          El::Output("Processed ", sdp.num_blocks, " SDP blocks in ",
                     (double)timer.timer().elapsed_milliseconds() / 1000,
                     " seconds, output: ", parameters.output_path.string());
        }

      if(parameters.verbosity >= Verbosity::debug)
        {
          timers.write_profile(parameters.output_path.string()
                               + ".profiling/profiling."
                               + std::to_string(El::mpi::Rank()));
        }
    }
  catch(std::exception &e)
    {
      std::cerr << "Error: " << e.what() << "\n" << std::flush;
      El::mpi::Abort(El::mpi::COMM_WORLD, 1);
    }
  catch(...)
    {
      std::cerr << "Unknown Error\n" << std::flush;
      El::mpi::Abort(El::mpi::COMM_WORLD, 1);
    }
}
