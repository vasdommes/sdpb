//#pragma once
//
//#include "Fmpz_Matrix.hxx"
//#include "Shared_Window_Array.hxx"
//#include "BigInt_Shared_Memory_Syrk_Context.hxx"
//
//void calculate_Block_Matrix_square(
//  El::mpi::Comm shared_memory_comm,
//  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
//  const std::vector<size_t> &block_indices_for_window,
//  Block_Residue_Matrices_Window<double> &blocks_window,
//  Residue_Matrices_Window<double> &output_residues_window, Fmpz_Comb &comb,
//  El::DistMatrix<El::BigFloat> &output);
//
//// Calculate Q := P^T P
////
//// P and Q are distributed among shared memory communicator, context.shared_memory_comm
//// (in practice - among all processes on a single machine).
////
//// bigint_input_matrix_blocks - horizontal bands of P matrix
//// bigint_output - Q matrix
////
//// Each process has some blocks of P, and their indices in communicator are stored in
//// block_indices_per_shared_memory_comm (vector of the same size as bigint_input_matrix_blocks)
////
//// Both P and Q elements should be (big) integers
//// We calculate residues of P modulo set of primes,
//// then multiply residue matrices via BLAS,
//// and restore Q from residues using Chinese Remainder Theorem
//void bigint_syrk_blas(
//  BigInt_Shared_Memory_Syrk_Context &context, El::UpperOrLower uplo,
//  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
//  const std::vector<size_t> &block_indices_per_shared_memory_comm,
//  El::DistMatrix<El::BigFloat> &bigint_output);
