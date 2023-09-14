#pragma once

#include "Fmpz_Matrix.hxx"
#include "Shared_Window_Array.hxx"

void calculate_Block_Matrix_square(
  El::DistMatrix<El::BigFloat> &output,
  // Blocks stored for a given rank
  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
  // Indices of input_normalized_blocks in blocks_window
  const std::vector<size_t> &block_indices_for_window,
  Block_Residue_Matrices_Window<double> &blocks_window,
  Residue_Matrices_Window<double> &result_window, Comb &comb);
