#pragma once

#include "pmp/Polynomial_Vector_Matrix.hxx"
#include "simpleboot/Linear_Combination_Of_Mathematica_Functions.hxx"
#include "simpleboot/PMP_Simpleboot_Parsing_Context.hxx"

#include <El.hpp>

#include <filesystem>
#include <optional>
#include <vector>

struct PMP_File_Parse_Result
{
public:
  using objective_type = std::optional<
    std::variant<std::vector<El::BigFloat>,
                 std::vector<Linear_Combination_Of_Mathematica_Functions>>>;
  using normalization_type = objective_type;

  // Vector a_0..a_N, see (3.1) in SDPB Manual
  objective_type objective;
  // Normalization vector n_0..n_N, see (3.1) in SDPB Manual
  normalization_type normalization;
  // Total number of PVM matrices in file
  size_t num_matrices = 0;
  // If file is read by several processes,
  // each process saves only some matrices, according to should_parse_matrix()
  // parsed_matrices is a map: index -> matrix,
  // where 0 <= index < num_matrices
  std::map<size_t, Polynomial_Vector_Matrix> parsed_matrices;

  PMP_File_Parse_Result() = default;

  static void validate(const PMP_File_Parse_Result &result);

  static PMP_File_Parse_Result
  read(const std::filesystem::path &input_path, int64_t max_num_poles,
       bool should_parse_objective, bool should_parse_normalization,
       const std::function<bool(size_t matrix_index)> &should_parse_matrix,
       const std::shared_ptr<PMP_Simpleboot_Parsing_Context<>>
         &simpleboot_context);

  // Allow moving and prevent accidential copying

  PMP_File_Parse_Result(const PMP_File_Parse_Result &other) = delete;
  PMP_File_Parse_Result(PMP_File_Parse_Result &&other) noexcept = default;
  PMP_File_Parse_Result &
  operator=(const PMP_File_Parse_Result &other) = delete;
  PMP_File_Parse_Result &
  operator=(PMP_File_Parse_Result &&other) noexcept = default;
};
