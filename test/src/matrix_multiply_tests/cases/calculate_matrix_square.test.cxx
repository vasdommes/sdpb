#include <catch2/catch_amalgamated.hpp>

#include "matrix_multiply/Fmpz_Comb.hxx"
#include "test_util/test_util.hxx"
#include "unit_tests/util/util.hxx"
#include "matrix_multiply/Shared_Window_Array.hxx"
#include "matrix_multiply/matrix_multiply.hxx"
#include "matrix_multiply/Matrix_Normalizer.hxx"

#include <vector>
#include <El.hpp>

using Test_Util::REQUIRE_Equal::diff;

namespace
{
  El::Matrix<El::BigFloat>
  calculate_matrix_square_El_Syrk(const El::Matrix<El::BigFloat> &P_matrix)
  {
    El::Matrix<El::BigFloat> Q_result_El_Syrk;
    El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
             El::BigFloat(1.0), P_matrix, Q_result_El_Syrk);
    El::MakeSymmetric(El::UpperOrLowerNS::UPPER, Q_result_El_Syrk);
    return Q_result_El_Syrk;
  }

  El::Matrix<El::BigFloat>
  calculate_matrix_square_El_Gemm(const El::Matrix<El::BigFloat> &P_matrix)
  {
    El::Matrix<El::BigFloat> Q_result_El_Gemm;
    El::Gemm(El::OrientationNS::TRANSPOSE, El::OrientationNS::NORMAL,
             El::BigFloat(1.0), P_matrix, P_matrix, Q_result_El_Gemm);
    return Q_result_El_Gemm;
  }

  // - Normalize, convert to fmpz (big integers)
  // - Calculate residues (mod a bunch of primes)
  // - Multiply matrices of residues via BLAS
  // - Restore the result using Chinese Remainder Theorem
  // - Convert back to BigFloat, restore
  El::Matrix<El::BigFloat> calculate_matrix_square_fmpz_mat_mul_blas(
    const El::Matrix<El::BigFloat> &P_matrix, int bits)
  {
    auto P = P_matrix;
    INFO("auto P = P_matrix;");
    CAPTURE(P);
    CAPTURE(bits);
    // matrix only on this rank, thus El::mpi::COMM_SELF, no communication
    Matrix_Normalizer normalizer(P, bits, El::mpi::COMM_SELF);

    normalizer.normalize_and_shift_P(P);

    El::Matrix<El::BigFloat> PT;
    Transpose(P, PT);
    INFO("Transpose(P, PT);");

    Fmpz_Matrix PT_bigint(PT);
    Fmpz_Matrix P_bigint(P);
    Fmpz_Matrix Q(P.Width(), P.Width());
    fmpz_mat_mul_blas(Q.fmpz_matrix, PT_bigint.fmpz_matrix,
                      P_bigint.fmpz_matrix);

    El::Matrix<El::BigFloat> Q_result;
    Q.ToBigFloatMatrix(Q_result);

    normalizer.restore_Q(Q_result);
    return Q_result;
  }
}

TEST_CASE("calculate_Block_Matrix_square")
{
  // input: dense tall NxK matrix P, splitted horizontally into blocks
  // output: NxN matrix Q := P^T * P

  MPI_Comm comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &comm);
  {
    INFO("TODO: the test currently doesn't work for several nodes");
    REQUIRE(El::mpi::Congruent(comm, El::mpi::COMM_WORLD));
  }

  // TODO fails with for loop in Release mode!
  // but works without for loop, or in Debug node
  for(size_t block_width : {1, 10})
    for(size_t num_blocks : {1, 10, 100, 500, 1000})
      //    for(size_t num_blocks : {100})
      DYNAMIC_SECTION("num_blocks=" << num_blocks
                                    << " block_width=" << block_width)
      //  size_t num_blocks=10; size_t block_width=5;
      {
        CAPTURE(num_blocks);
        CAPTURE(block_width);

        std::vector<size_t> block_heights(num_blocks);
        size_t total_block_height = 0;
        for(size_t block_index = 0; block_index < num_blocks; ++block_index)
          {
            size_t height = block_index + 100;
            //            size_t height = 1;
            block_heights.at(block_index) = height;
            total_block_height += height;
          }
        CAPTURE(total_block_height);

        int bits;
        CAPTURE(bits = El::gmp::Precision());
        int diff_precision;
        CAPTURE(diff_precision = bits / 2);

        // P_matrix is a tall matrix of all blocks.
        // We initialize it on rank=0 and then copy to all ranks.
        El::Matrix<El::BigFloat> P_matrix(total_block_height, block_width);
        El::Matrix<El::BigFloat> Q_result_El_Syrk(block_width, block_width);
        if(El::mpi::Rank() == 0)
          {
            P_matrix
              = Test_Util::random_matrix(total_block_height, block_width);

            // Fill with 1.0 - for easier debug:
            // El::Fill(P_matrix, El::BigFloat(1.0));

            Q_result_El_Syrk = calculate_matrix_square_El_Syrk(P_matrix);

            // Double-check our result with Gemm
            auto Q_result_El_Gemm = calculate_matrix_square_El_Gemm(P_matrix);
            DIFF(Q_result_El_Syrk, Q_result_El_Gemm);

            // Check fmpz_mat_mul_blas, which is essentially
            // the same as our method.
            auto Q_result_fmpz_mat_mul_blas
              = calculate_matrix_square_fmpz_mat_mul_blas(P_matrix, bits);
            DIFF_PREC(Q_result_El_Syrk, Q_result_fmpz_mat_mul_blas,
                      diff_precision);
            INFO("after DIFF_PREC(Q_result_El_Syrk, "
                 "Q_result_fmpz_mat_mul_blas, diff_precision);");
          }

        //
        El::mpi::Broadcast(P_matrix.Buffer(), P_matrix.MemorySize(), 0,
                           El::mpi::COMM_WORLD);
        // Send copies of Q_result_El_Syrk to all ranks, to make comparison easy
        // TODO use DistMatrix instead?
        El::mpi::Broadcast(Q_result_El_Syrk.Buffer(),
                           Q_result_El_Syrk.MemorySize(), 0,
                           El::mpi::COMM_WORLD);
        //        CAPTURE(P_matrix);
        CAPTURE(Q_result_El_Syrk);

        // Setup blocks for FLINT+BLAS multiplication

        std::vector<El::DistMatrix<El::BigFloat>> blocks;
        std::vector<size_t> block_indices;

        int global_block_offset = 0;
        for(size_t block_index = 0; block_index < block_heights.size();
            ++block_index)
          {
            int block_height = block_heights.at(block_index);

            if(block_index % El::mpi::Size(comm) == El::mpi::Rank(comm))
              {
                // TODO test for DistMatrix distributed over several ranks
                El::DistMatrix<El::BigFloat> block(block_height, block_width,
                                                   El::Grid::Trivial());

                for(int i = 0; i < block.LocalHeight(); ++i)
                  for(int j = 0; j < block.LocalWidth(); ++j)
                    {
                      int global_row
                        = block.GlobalRow(i) + global_block_offset;
                      int global_col = block.GlobalCol(j);
                      block.SetLocal(i, j,
                                     P_matrix.Get(global_row, global_col));
                    }
                // TODO block indices for window!
                block_indices.push_back(block_index);
                blocks.push_back(block);
              }
            global_block_offset += block_height;
          }
        //  CAPTURE(blocks.at(0).Get(0, 0));
        //  CAPTURE(blocks.at(0).Get(0, 0) * blocks.at(0).Get(0, 0));

        const El::Grid result_grid(comm);

        // calculate via BLAS

        int sign = 1;
        Fmpz_Comb comb(bits, bits, sign, total_block_height);
        CAPTURE(comb.num_primes);
        CAPTURE(comb.primes);
        CAPTURE(FLINT_BIT_COUNT(total_block_height));

        // blocks are distributed among all ranks
        Matrix_Normalizer normalizer(blocks, block_width, bits,
                                     El::mpi::COMM_WORLD);
        CAPTURE(normalizer.column_norms);
        for(auto &block : blocks)
          {
            //        El::Output(El::mpi::Rank(), "normalize_and_shift");
            normalizer.normalize_and_shift_P(block);
          }

        //  CAPTURE(blocks[0].Get(0, 0));

        // TODO test normalization

        // TODO grid
        //  auto& result_grid = El::Grid::Default();
        //  const El::Grid result_grid(El::mpi::COMM_WORLD);

        El::DistMatrix<El::BigFloat> Q_result(block_width, block_width,
                                              result_grid);

        Block_Residue_Matrices_Window<double> block_residues_window(
          comm, comb.num_primes, block_heights.size(), block_heights,
          block_width);
        Residue_Matrices_Window<double> result_residues_window(
          comm, comb.num_primes, block_width, block_width);

        calculate_Block_Matrix_square(comm, blocks, block_indices,
                                      block_residues_window,
                                      result_residues_window, comb, Q_result);

        CAPTURE(num_blocks);
        CAPTURE(block_width);
        if(El::mpi::Rank() == 0)
          {
            //            std::vector<double> block_residues(
            //              block_residues_window.residues[0].LockedBuffer(),
            //              block_residues_window.residues[0].LockedBuffer()
            //                + primes.size() * num_blocks);
            //
            //            std::vector<double> result_residues(
            //              result_residues_window.residues[0].LockedBuffer(),
            //              result_residues_window.residues[0].LockedBuffer()
            //                + primes.size());
            //
            std::vector<double> result_00_residues;
            double max_Q_residue = 0;
            for(const auto &matrix : result_residues_window.residues)
              {
                double residue = matrix.Get(0, 0);
                result_00_residues.emplace_back(residue);
                max_Q_residue = std::max(residue, max_Q_residue);
              }

            CAPTURE(result_00_residues);

            CAPTURE(comb.primes.at(0));
            auto max_uint32_P_residue = comb.primes.at(0) / 2;
            CAPTURE(max_uint32_P_residue * max_uint32_P_residue
                    * total_block_height);
            CAPTURE(max_Q_residue);
            REQUIRE((double)max_uint32_P_residue * max_uint32_P_residue
                      * total_block_height
                    >= max_Q_residue);
            CAPTURE(std::numeric_limits<uint32_t>::max());
            CAPTURE(std::numeric_limits<slong>::max());

            CAPTURE(Q_result.Matrix());
            // FAIL();
          }

        //  INFO("normshifted Q_result:");
        //  CAPTURE(Q_result);
        //  INFO("restore...");
        normalizer.restore_Q(Q_result);
        CAPTURE(Q_result);

        for(int i = 0; i < Q_result.LocalHeight(); ++i)
          for(int j = 0; j < Q_result.LocalWidth(); ++j)
            {
              CAPTURE(El::mpi::Rank());
              CAPTURE(i);
              CAPTURE(j);
              auto global_row = Q_result.GlobalRow(i);
              auto global_col = Q_result.GlobalCol(j);
              CAPTURE(global_row);
              CAPTURE(global_col);

              CAPTURE(El::gmp::Precision());
              CAPTURE(FLINT_BIT_COUNT(total_block_height));

              DIFF_PREC(Q_result.GetLocal(i, j),
                        Q_result_El_Syrk.Get(global_row, global_col),
                        diff_precision);
            }
      }
}
