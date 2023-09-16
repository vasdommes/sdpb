#include <catch2/catch_amalgamated.hpp>

#include "matrix_multiply/Fmpz_Comb.hxx"
#include "test_util/test_util.hxx"
#include "unit_tests/util/util.hxx"
#include "matrix_multiply/Matrix_Normalizer.hxx"

#include <El.hpp>

#include <random>

using Test_Util::REQUIRE_Equal::diff;

TEST_CASE("normalize_and_shift")
{
  int height = GENERATE(1, 10, 100);

  DYNAMIC_SECTION("height=" << height)
  {
    int bits = El::gmp::Precision();

    // width is not really important,
    // we just create non-square matrix for generality
    int width = height / 2 + 1;

    // non-square matrix for
    El::DistMatrix<El::BigFloat> P_matrix
      = Test_Util::random_distmatrix(height, width);
    CAPTURE(P_matrix.Height());
    CAPTURE(P_matrix.Width());
    CAPTURE(P_matrix.LocalHeight());
    CAPTURE(P_matrix.LocalWidth());

    std::vector<El::DistMatrix<El::BigFloat>> P_matrix_blocks;

    std::default_random_engine rand_engine;

    // Split P_matrix horizontally
    // into blocks with different random heights
    int begin_row = 0;
    while(begin_row < P_matrix.Height())
      {
        std::uniform_int_distribution<int> dist(begin_row + 1,
                                                P_matrix.Height());
        int end_row = dist(rand_engine);
        assert(end_row > begin_row);
        assert(end_row <= P_matrix.Height());

        El::Range<int> rows(begin_row, end_row);
        El::Range<int> cols(0, P_matrix.Width());
        P_matrix_blocks.emplace_back(P_matrix(rows, cols));

        begin_row = end_row;
      }
    CAPTURE(P_matrix_blocks.size());

    auto initial_P_matrix = P_matrix;

    Matrix_Normalizer normalizer(P_matrix_blocks, width, bits,
                                 El::mpi::COMM_WORLD);
    CAPTURE(normalizer.precision);
    CAPTURE(normalizer.column_norms);
    normalizer.normalize_and_shift_P(P_matrix);

    SECTION("normalize_and_shift_P")
    {
      for(int i = 0; i < P_matrix.LocalHeight(); ++i)
        for(int j = 0; j < P_matrix.LocalWidth(); ++j)
          {
            CAPTURE(i);
            CAPTURE(j);

            auto initial_value = initial_P_matrix.GetLocal(i, j);
            auto normshifted_value = P_matrix.GetLocal(i, j);
            CAPTURE(initial_value);
            CAPTURE(normshifted_value);

            const auto &norm
              = normalizer.column_norms.at(P_matrix.GlobalCol(j));
            CAPTURE(P_matrix.GlobalCol(j));
            CAPTURE(norm);

            auto restored_value = (normshifted_value >> bits) * norm;
            DIFF(initial_value, restored_value);
          }
    }

    SECTION("restore_Q")
    {
      INFO(
        "Check that calculating Q with and without intermediate normalization "
        "gives the same result up to a reasonable diff_precision");
      int diff_precision;
      CAPTURE(diff_precision = bits / 2);

      El::DistMatrix<El::BigFloat> initial_Q, Q;
      El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
               El::BigFloat(1), initial_P_matrix, initial_Q);
      El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
               El::BigFloat(1), P_matrix, Q);

      {
        INFO("Check that normalized matrix squared has 1.0 on diagonal");
        for(int i = 0; i < P_matrix.LocalHeight(); ++i)
          for(int j = 0; j < P_matrix.LocalWidth(); ++j)
            {
              if(Q.GlobalRow(i) == Q.GlobalCol(j))
                DIFF_PREC(Q.GetLocal(i, j) >> 2 * bits, El::BigFloat(1.0),
                          diff_precision);
            }
      }

      normalizer.restore_Q(Q);

      DIFF_PREC(initial_Q, Q, diff_precision);
    }
  }
}