#pragma once

#include "Fmpz_Matrix.hxx"
#include "Shared_Window_Array.hxx"

void calculate_Block_Matrix_square(
  El::mpi::Comm shared_memory_comm,
  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
  const std::vector<size_t> &block_indices_for_window,
  Block_Residue_Matrices_Window<double> &blocks_window,
  Residue_Matrices_Window<double> &output_residues_window, Fmpz_Comb &comb,
  El::DistMatrix<El::BigFloat> &output);
