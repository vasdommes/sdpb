#include <catch2/catch_amalgamated.hpp>

#include "matrix_multiply/Normalized_Matrix.hxx"
#include "matrix_multiply/Primes.hxx"
#include "test_util/diff.hxx"
#include "unit_tests/util/util.hxx"
#include "matrix_multiply/Shared_Window_Array.hxx"
#include "matrix_multiply/matrix_multiply.hxx"
#include "sdp_solve/Block_Info.hxx"

#include <vector>
#include <El.hpp>

using Test_Util::REQUIRE_Equal::diff;

namespace
{
  void El_calculate_matrix_square(
    El::DistMatrix<El::BigFloat> &output,
    const std::vector<El::DistMatrix<El::BigFloat>> &input_blocks,
    int block_width)
  {
    REQUIRE(output.Height() == block_width);
    REQUIRE(output.Width() == block_width);

    // TODO account for other grids

    El::DistMatrix<El::BigFloat> Q_group(block_width, block_width,
                                         El::Grid::Trivial());
    if(input_blocks.size() > 0)
      Q_group.SetGrid(input_blocks.at(0).Grid());

    // See initialize_Q_group
    for(const auto &block : input_blocks)
      {
        REQUIRE(block_width == block.Width());
        REQUIRE(block.Grid() == Q_group.Grid());
        //      El::DistMatrix<El::BigFloat> Q_group_view(
        //          El::View(Q_group, 0, 0, block_width, block_width));
        //        El::Syrk(El::UpperOrLowerNS::UPPER,
        //        El::OrientationNS::TRANSPOSE,
        //                 El::BigFloat(1), block,
        //                 El::BigFloat(1), Q_group_view);
        El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
                 El::BigFloat(1), block, El::BigFloat(1), Q_group);
      }

    for(int i = 0; i < Q_group.Height(); ++i)
      for(int j = 0; j < Q_group.Width(); ++j)
        {
          //          int global_row = Q_group.GlobalRow(i);
          //          int global_col = Q_group.GlobalCol(j);
          //          El::mpi::Barrier();
          // TODO synchronize corectly!
          El::BigFloat value = Q_group.IsLocal(i, j) ? Q_group.Get(i, j) : 0;
          value
            = El::mpi::Reduce(value, output.Owner(i, j), El::mpi::COMM_WORLD);

          output.Update(i, j, value);

          El::mpi::Barrier();
        }
  }
}

namespace
{
  // For each column:
  // Sum of squares of elements stored in this rank
  void
  add_local_column_norms_squared(std::vector<El::BigFloat> &local_norms_squared,
                                 const El::DistMatrix<El::BigFloat> &matrix)
  {
    assert(local_norms_squared.size() == matrix.Width());
    for(int i = 0; i < matrix.LocalHeight(); ++i)
      for(int j = 0; j < matrix.LocalWidth(); ++j)
        {
          auto value = matrix.GetLocal(i, j);
          int global_col = matrix.GlobalCol(j);
          local_norms_squared.at(global_col) += value * value;
        }
  }

  std::vector<El::BigFloat> calculate_column_norms(
    const std::vector<El::DistMatrix<El::BigFloat>> &input_blocks,
    size_t block_width)
  {
    // For each column,
    // we calculate norm squared for each block
    // and accumulate them for all blocks from all ranks

    std::vector<El::BigFloat> local_norms_squared(block_width, 0);
    std::vector<El::BigFloat> column_norms(block_width, 0);

    for(const auto &input_block : input_blocks)
      {
        add_local_column_norms_squared(local_norms_squared, input_block);
      }
    // TODO do we need a barrier before AllReduce? probably no, check
    El::mpi::Barrier();

    // TODO reduce the whole vector at once; maybe reuse El::Matrix for that?
    for(size_t i = 0; i < block_width; ++i)
      {
        auto norm_squared
          = El::mpi::AllReduce(local_norms_squared.at(i), El::mpi::COMM_WORLD);
        column_norms.at(i) = Sqrt(norm_squared);
      }
    return column_norms;
  }

  // normalize coulmns and multiply by 2^N
  void normalize_and_shift(El::DistMatrix<El::BigFloat> &matrix,
                           const std::vector<El::BigFloat> &column_norms,
                           int bit_shift)
  {
    for(int j = 0; j < matrix.LocalWidth(); ++j)
      {
        int global_col = matrix.GlobalCol(j);
        const auto &norm = column_norms.at(global_col);
        if(norm == El::BigFloat(0))
          continue;
        for(int i = 0; i < matrix.LocalHeight(); ++i)
          {
            auto normalized_value = matrix.GetLocal(i, j) / norm;
            matrix.SetLocal(i, j, normalized_value << bit_shift);
          }
      }
  }

  //  void normalize_and_shift(std::vector<El::DistMatrix<El::BigFloat>>
  //  &blocks,
  //                           size_t block_width, int bit_shift)
  //  {
  //    auto column_norms = calculate_column_norms(blocks, block_width);
  //    for(auto &block : blocks)
  //      {
  //        normalize_and_shift(block, column_norms, bit_shift);
  //      }
  //  }

  // restore Q := P^T * P
  // from Q' := P'^T * P'
  // TODO here we shift by 2*bit_shift, this is not clear from signature!
  void restore(El::DistMatrix<El::BigFloat> &Q,
               const std::vector<El::BigFloat> &column_norms, int bit_shift)
  {
    for(int i = 0; i < Q.LocalHeight(); ++i)
      for(int j = 0; j < Q.LocalWidth(); ++j)
        {
          int global_row = Q.GlobalRow(i);
          int global_col = Q.GlobalCol(j);
          auto value = Q.GetLocal(i, j);
          value >>= 2 * bit_shift;
          value *= column_norms.at(global_row) * column_norms.at(global_col);
          Q.SetLocal(i, j, value);
        }
  }
}

namespace Catch
{
  template <> struct StringMaker<El::Matrix<El::BigFloat>>
  {
    static std::string convert(const El::Matrix<El::BigFloat> &matrix)
    {
      std::ostringstream os;
      for(int i = 0; i < matrix.Height(); ++i)
        {
          os << "\n";
          for(int j = 0; j < matrix.Width(); ++j)
            {
              os << " " << matrix.Get(i, j);
            }
        }
      return os.str();
    }
  };
  // TODO deduplicate
  template <> struct StringMaker<El::DistMatrix<El::BigFloat>>
  {
    static std::string convert(const El::DistMatrix<El::BigFloat> &matrix)
    {
      std::ostringstream os;
      for(int i = 0; i < matrix.Height(); ++i)
        {
          os << "\n";
          for(int j = 0; j < matrix.Width(); ++j)
            {
              os << " " << matrix.Get(i, j);
            }
        }
      return os.str();
    }
  };
}

TEST_CASE("normalize_and_shift")
{
  int N = GENERATE(1, 10, 100);

  DYNAMIC_SECTION("N=" << N)
  {
    El::DistMatrix<El::BigFloat> matrix(N, N);
    for(int i = 0; i < matrix.LocalHeight(); ++i)
      for(int j = 0; j < matrix.LocalWidth(); ++j)
        {
          matrix.SetLocal(i, j, Test_Util::random_bigfloat());
        }

    auto initial_matrix = matrix;

    int bits = El::gmp::Precision();
    auto column_norms = calculate_column_norms({matrix}, matrix.Width());
    normalize_and_shift(matrix, column_norms, bits);
    for(int i = 0; i < matrix.LocalHeight(); ++i)
      for(int j = 0; j < matrix.LocalWidth(); ++j)
        {
          auto initial_value = initial_matrix.GetLocal(i, j);
          auto normshifted_value = matrix.GetLocal(i, j);
          CAPTURE(normshifted_value);
          auto norm = column_norms.at(matrix.GlobalCol(j));

          auto restored_value = (normshifted_value >> bits) * norm;
          DIFF(initial_value, restored_value);
        }
  }
}

TEST_CASE("calculate_Block_Matrix_square")
{
  // input: dense tall NxK matrix P, splitted horizontally into blocks
  // output: NxN matrix Q := P^T * P

  // Total number of blocks in the problem
  //  size_t num_blocks = GENERATE(1, 10);
  //  size_t block_width = GENERATE(1, 5);
  //

  // TODO fails with for loop in Release mode!
  // but works without for loop, or in Debug node
  for(size_t num_blocks : {2, 3})
    for(size_t block_width : {5})
      //  DYNAMIC_SECTION("num_blocks=" << num_blocks << " block_width=" <<
      //  block_width) size_t num_blocks=10; size_t block_width=5;
      {
        CAPTURE(num_blocks);
        CAPTURE(block_width);

        MPI_Comm comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                            MPI_INFO_NULL, &comm);
        {
          INFO("TODO: the test currently doesn't work for several nodes");
          REQUIRE(El::mpi::Congruent(comm, El::mpi::COMM_WORLD));
        }

        std::vector<size_t> block_heights(num_blocks);
        size_t total_block_height = 0;
        for(size_t block_index = 0; block_index < num_blocks; ++block_index)
          {
            size_t height = block_index + 100;
            //            size_t height = 1;
            block_heights.at(block_index) = height;
            total_block_height += height;
          }

        // P_matrix is a tall matrix of all blocks.
        // We initialize it on rank=0 and then copy to all ranks.
        El::Matrix<El::BigFloat> P_matrix(total_block_height, block_width);
        El::Matrix<El::BigFloat> Q_result_El_Syrk(block_width, block_width);
        if(El::mpi::Rank() == 0)
          {
            P_matrix
              = Test_Util::random_matrix(total_block_height, block_width);
            //       TODO 1.0 - for easier debug:
            //      P_matrix = Test_Util::random_matrix(total_block_height,
            //      block_width,
            //                                          []() { return 1.0; });

            El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
                     El::BigFloat(1.0), P_matrix, Q_result_El_Syrk);
            // Initialize lower half:
            for(int i = 0; i < Q_result_El_Syrk.Height(); ++i)
              for(int j = 0; j < i; ++j)
                Q_result_El_Syrk(i, j) = Q_result_El_Syrk(j, i);

            // Double-check our result with Gemm
            El::Matrix<El::BigFloat> Q_result_El_Gemm(block_width,
                                                      block_width);
            El::Gemm(El::OrientationNS::TRANSPOSE, El::OrientationNS::NORMAL,
                     El::BigFloat(1.0), P_matrix, P_matrix, Q_result_El_Gemm);

            DIFF(Q_result_El_Syrk, Q_result_El_Gemm);
          }

        //
        El::mpi::Broadcast(P_matrix.Buffer(), P_matrix.MemorySize(), 0,
                           El::mpi::COMM_WORLD);
        // Send copies of Q_result_El_Syrk to all ranks, to make comparison easy
        // TODO use DistMatrix instead?
        El::mpi::Broadcast(Q_result_El_Syrk.Buffer(),
                           Q_result_El_Syrk.MemorySize(), 0,
                           El::mpi::COMM_WORLD);
        CAPTURE(P_matrix);
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

        int bits = El::gmp::Precision();
        int sign = 1;
        Primes primes(bits, bits, sign, total_block_height);

        //    normalize_and_shift(blocks, block_width, bits);
        auto column_norms = calculate_column_norms(blocks, block_width);
        for(auto &block : blocks)
          {
            //        El::Output(El::mpi::Rank(), "normalize_and_shift");
            normalize_and_shift(block, column_norms, bits);
          }

        CAPTURE(column_norms[0]);

        //  CAPTURE(blocks[0].Get(0, 0));

        // TODO test normalization

        // TODO grid
        //  auto& result_grid = El::Grid::Default();
        //  const El::Grid result_grid(El::mpi::COMM_WORLD);

        El::DistMatrix<El::BigFloat> Q_result(block_width, block_width,
                                              result_grid);

        Block_Residue_Matrices_Window<double> block_residues_window(
          comm, primes.size(), block_heights.size(), block_heights,
          block_width);
        Residue_Matrices_Window<double> result_residues_window(
          comm, primes.size(), block_width, block_width);

        calculate_Block_Matrix_square(Q_result, blocks, block_indices,
                                      block_residues_window,
                                      result_residues_window, primes);

        CAPTURE(num_blocks);
        CAPTURE(block_width);
        if(El::mpi::Rank() == 0)
          {
            std::vector<double> block_residues(
              block_residues_window.residues[0].LockedBuffer(),
              block_residues_window.residues[0].LockedBuffer()
                + primes.size() * num_blocks);

            std::vector<double> result_residues(
              result_residues_window.residues[0].LockedBuffer(),
              result_residues_window.residues[0].LockedBuffer()
                + primes.size());
            CAPTURE(result_residues);
            CAPTURE(Q_result.Matrix());
            REQUIRE(Q_result.GetLocal(0, 0) != El::BigFloat(0));
            //      FAIL();
          }

        //  INFO("normshifted Q_result:");
        //  CAPTURE(Q_result);
        //  INFO("restore...");
        restore(Q_result, column_norms, bits);
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
              DIFF_PREC(Q_result.GetLocal(i, j),
                        Q_result_El_Syrk.Get(global_row, global_col),
                        El::gmp::Precision() / 2 - FLINT_BIT_COUNT(block_width)
                          - 1);
            }
      }
}
