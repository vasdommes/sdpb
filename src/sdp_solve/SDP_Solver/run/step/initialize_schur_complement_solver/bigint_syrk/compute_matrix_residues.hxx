#pragma once

#include <El.hpp>
#include <flint/nmod.h>
#include "Fmpz_Comb.hxx"
#include "Shared_Window_Array.hxx"
#include "fmpz_mul_blas_util.hxx"
#include "fmpz_BigFloat_convert.hxx" //TODO join with prev header

// compute residues and put them to shared window
// NB: input_block is BigInt matrix (normalized matrix, multiplied by 2^N)
inline void compute_matrix_residues(
  size_t block_index_in_node,
  const El::DistMatrix<El::BigFloat> &bigint_input_block, Fmpz_Comb &comb,
  Block_Residue_Matrices_Window<double> &block_residues_window)
{
  assert(block_residues_window.width == bigint_input_block.Width());

  // for each input_matrix element
  // for each prime_index=0..primes.size():
  // - Calculate input_matrix(i,j) mod primes[prime_index]
  // - Write the result to output_window, with
  //   offset = start_offset + prime_index * prime_stride
  fmpz_t bigint_value;
  fmpz_init(bigint_value);
  for(int i = 0; i < bigint_input_block.Height(); ++i)
    for(int j = 0; j < bigint_input_block.Width(); ++j)
      {
        if(bigint_input_block.IsLocal(i, j))
          {
            // pointer to the first residue
            double *data = block_residues_window.block_residues.at(0)
                             .at(block_index_in_node)
                             .Buffer(i, j);
            BigFloat_to_fmpz_t(bigint_input_block.Get(i, j), bigint_value);
            fmpz_multi_mod_uint32_stride(
              data, block_residues_window.prime_stride, bigint_value, comb);
          }
      }
  fmpz_clear(bigint_value); // TODO wrap with RAII
}
