#pragma once
#include "pmp/Polynomial_Matrix_Program.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "simpleboot/data_provider/Simpleboot_Parameters.hxx"

#include <filesystem>
#include <vector>

Polynomial_Matrix_Program read_polynomial_matrix_program(
  const Environment &env,
  const std::vector<std::filesystem::path> &input_files, int64_t max_num_poles,
  const Verbosity &verbosity, Timers &timers,
  const std::optional<Simpleboot_Parameters> &simpleboot_parameters
  = std::nullopt);

Polynomial_Matrix_Program
read_polynomial_matrix_program(const Environment &env,
                               const std::filesystem::path &input_file,
                               int64_t max_num_poles,
                               const Verbosity &verbosity, Timers &timers);

std::vector<std::filesystem::path>
read_nsv_file_list(const std::filesystem::path &input_file);

std::vector<std::filesystem::path>
collect_files_expanding_nsv(const std::filesystem::path &input_file);

std::vector<std::filesystem::path> collect_files_expanding_nsv(
  const std::vector<std::filesystem::path> &input_files);
