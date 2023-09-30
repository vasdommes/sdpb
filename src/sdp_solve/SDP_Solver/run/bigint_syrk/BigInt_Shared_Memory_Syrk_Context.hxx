#pragma once
#include "Shared_Window_Array.hxx"
#include "Fmpz_Matrix.hxx"
#include "Fmpz_Comb.hxx"
#include "../../../../Timers.hxx"

// TODO rename
struct BigInt_Shared_Memory_Syrk_Context : boost::noncopyable
{
  El::mpi::Comm shared_memory_comm;
  Fmpz_Comb comb;
  Block_Residue_Matrices_Window<double> input_block_residues_window;
  Residue_Matrices_Window<double> output_residues_window;
  const std::vector<size_t> block_index_local_to_shmem;

  // block_heights - for all blocks in shared memory
  BigInt_Shared_Memory_Syrk_Context(
    const El::mpi::Comm &shared_memory_comm, mp_bitcnt_t precision,
    const std::vector<El::Int> &block_heights, El::Int block_width,
    const std::vector<size_t> &block_index_local_to_shmem);

  // Calculate Q := P^T P
  //
  // P and Q are distributed among shared memory communicator, context.shared_memory_comm
  // (in practice - among all processes on a single machine).
  //
  // bigint_input_matrix_blocks - horizontal bands of P matrix
  // bigint_output - Q matrix
  //
  // Each process has some blocks of P, and their indices in communicator are stored in
  // block_indices_per_shared_memory_comm (vector of the same size as bigint_input_matrix_blocks)
  //
  // Both P and Q elements should be (big) integers;
  // We calculate residues of P modulo set of primes,
  // then multiply residue matrices via BLAS,
  // and restore Q from residues using Chinese Remainder Theorem
  //
  // If you want to square arbitrary BigFloat matrix P,
  // then use Matrix_Normalizer before and after calling this bigint_syrk_blas()
  void bigint_syrk_blas(
    El::UpperOrLower uplo,
    const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
    El::DistMatrix<El::BigFloat> &bigint_output, Timers &timers);
};
