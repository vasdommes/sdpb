#include <catch2/catch_amalgamated.hpp>

#include "matrix_multiply/Fmpz_Matrix.hxx"
#include "matrix_multiply/Normalized_Matrix.hxx"
#include "set_stream_precision.hxx"
#include "test_util/diff.hxx"
#include "unit_tests/util/util.hxx"

#include <iostream>
#include <vector>
#include <gmp.h>
#include <gmpxx.h>
#include <cblas.h>
#include <chrono>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/ulong_extras.h>
#include <flint/nmod.h>

#include <El.hpp>
#include <boost/multiprecision/mpfr.hpp>

using Test_Util::REQUIRE_Equal::diff;

// code copied from flint mul_blas.c
namespace Primes
{
#define MAX_BLAS_DP_INT (UWORD(1) << 53)

  static mp_limb_t *
  _calculate_primes(slong *num_primes_, flint_bitcnt_t bits, slong k)
  {
    slong num_primes, primes_alloc;
    mp_limb_t *primes;
    mp_limb_t p;
    fmpz_t prod;

    p = 2 + 2 * n_sqrt((MAX_BLAS_DP_INT - 1) / (ulong)k);
    if(bits > 200)
      {
        /* if mod is the bottleneck, ensure p1*p2*p3 < 2^62 */
        p = FLINT_MIN(p, UWORD(1664544));
      }

    primes_alloc = 1 + bits / FLINT_BIT_COUNT(p);
    primes = FLINT_ARRAY_ALLOC(primes_alloc, mp_limb_t);
    num_primes = 0;

    fmpz_init_set_ui(prod, 1);

    do
      {
        do
          {
            if(p < 1000)
              {
                fmpz_clear(prod);
                flint_free(primes);
                *num_primes_ = 0;
                return NULL;
              }
            p--;
        } while(!n_is_prime(p));

        if(num_primes + 1 > primes_alloc)
          {
            primes_alloc = FLINT_MAX(num_primes + 1, primes_alloc * 5 / 4);
            primes = FLINT_ARRAY_REALLOC(primes, primes_alloc, mp_limb_t);
          }

        primes[num_primes] = p;
        num_primes++;

        fmpz_mul_ui(prod, prod, p);

    } while(fmpz_bits(prod) <= bits);

    fmpz_clear(prod);

    *num_primes_ = num_primes;
    return primes;
  }

  std::vector<mp_limb_t> calculate_primes(flint_bitcnt_t bits, slong k)
  {
    slong n;
    auto primes = _calculate_primes(&n, bits, k);
    return std::vector<mp_limb_t>(primes, primes + n);
  }

  static uint32_t _reduce_uint32(mp_limb_t a, nmod_t mod)
  {
    mp_limb_t r;
    NMOD_RED(r, a, mod);
    return (uint32_t)r;
  }

  static std::vector<uint32_t> fmpz_multi_mod_uint32_stride(
    //    uint32_t * out,
    //    slong stride,
    const fmpz_t input, const fmpz_comb_t C, fmpz_comb_temp_t CT)
  {
    std::vector<uint32_t> out(C->num_primes);
    slong stride = 1;

    slong i, k, l;
    fmpz *A = CT->A;
    mod_lut_entry *lu;
    slong *offsets;
    slong klen = C->mod_klen;
    fmpz_t ttt;

    /* high level split */
    if(klen == 1)
      {
        *ttt = A[0];
        A[0] = *input;
      }
    else
      {
        _fmpz_multi_mod_precomp(A, C->mod_P, input, -1, CT->T);
      }

    offsets = C->mod_offsets;
    lu = C->mod_lu;

    for(k = 0, i = 0, l = 0; k < klen; k++)
      {
        slong j = offsets[k];

        for(; i < j; i++)
          {
            /* mid level split: depends on FMPZ_MOD_UI_CUTOFF */
            mp_limb_t t = fmpz_get_nmod(A + k, lu[i].mod);

            /* low level split: 1, 2, or 3 small primes */
            if(lu[i].mod2.n != 0)
              {
                FLINT_ASSERT(l + 3 <= C->num_primes);
                out[l * stride] = _reduce_uint32(t, lu[i].mod0);
                l++;
                out[l * stride] = _reduce_uint32(t, lu[i].mod1);
                l++;
                out[l * stride] = _reduce_uint32(t, lu[i].mod2);
                l++;
              }
            else if(lu[i].mod1.n != 0)
              {
                FLINT_ASSERT(l + 2 <= C->num_primes);
                out[l * stride] = _reduce_uint32(t, lu[i].mod0);
                l++;
                out[l * stride] = _reduce_uint32(t, lu[i].mod1);
                l++;
              }
            else
              {
                FLINT_ASSERT(l + 1 <= C->num_primes);
                out[l * stride] = (uint32_t)(t);
                l++;
              }
          }
      }

    FLINT_ASSERT(l == C->num_primes);

    if(klen == 1)
      A[0] = *ttt;

    return out;
  }

  //  void init_comb(fmpz_comb_t comb, const std::vector<mp_limb_t> &primes)
  //  {
  //    mp_limb_signed_t num_primes = primes.size();
  //    fmpz_comb_init(comb, primes.data(), num_primes);
  //  }

  // In SDPB code, we'll have a bigint matrix
  // TODO: currently we start with a DistMatrix<BigFloat>!
  // We convert it to ordinary Fmpz_Matrix
  // We want to calculate residues and write them to a shared memory window,
  // which is an array of doubles
  // Get_value(window, prime, block, i,j) := window[ j + width*i + width*sum(block heights for block_id < block) + width*sum(block heights) * prime]
  //=> for fmpz_multi_mod_uint32_stride: stride = width*sum(block heights)
  std::vector<El::Matrix<double>>
  fmpz_mat_residues(const std::vector<mp_limb_t> &primes, const fmpz_mat_t x)
  {
    fmpz_comb_t comb;
    mp_limb_signed_t num_primes = primes.size();
    fmpz_comb_init(comb, primes.data(), num_primes);

    fmpz_comb_temp_t comb_temp;

    fmpz_comb_temp_init(comb_temp, comb);

    std::vector<El::Matrix<double>> x_mod(num_primes);
    for(auto &mat : x_mod)
      {
        mat.Resize(x->r, x->c);
      }

    for(int i = 0; i < x->r; ++i)
      for(int j = 0; j < x->c; ++j)
        {
          auto mods = fmpz_multi_mod_uint32_stride(fmpz_mat_entry(x, i, j),
                                                   comb, comb_temp);
          for(size_t p = 0; p < primes.size(); ++p)
            {
              // TODO write to a shared memory window
              x_mod[p].Set(i, j, (double)mods[p]);
            }
        }
    return x_mod;
  }

  El::Matrix<double>
  El_multiply_noblas(const El::Matrix<double> &x, const El::Matrix<double> &y)
  {
    El::Matrix<double> result;
    El::Gemm(El::OrientationNS::NORMAL, El::OrientationNS::NORMAL, 1.0, x, y,
             result);
    return result;
  }

  El::Matrix<double>
  El_multiply_blas(const El::Matrix<double> &x, const El::Matrix<double> &y)
  {
    int M = x.Height();
    int N = y.Width();
    int K = x.Width();
    El::Matrix<double> result(M, N);

    const double alpha = 1.0;
    const double beta = 0.0;
    const double* A = x.LockedBuffer();
    const double* B = y.LockedBuffer();
    double* C = result.Buffer();
    int lda = x.LDim();
    int ldb = y.LDim();
    int ldc = result.LDim();
    //  C := alpha*op(A)*op(B) + beta*C
    // where:
    //
    // op(X) is one of op(X) = X, or op(X) = XT, or op(X) = XH,
    //
    // alpha and beta are scalars,
    //
    // A, B and C are matrices:
    //
    // op(A) is an m-by-k matrix,
    //
    // op(B) is a k-by-n matrix,
    //
    // C is an m-by-n matrix.
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A,
                lda, B, ldb, beta, C, ldc);
    return result;
  }

  El::Matrix<uint32_t>
  to_uint32_matrix(mp_limb_t prime, const El::Matrix<double> &matrix)
  {
    El::Matrix<uint32_t> result(matrix.Height(), matrix.Width());

    nmod_t mod;
    nmod_init(&mod, prime);
    ulong shift = ((2 * MAX_BLAS_DP_INT) / mod.n) * mod.n;

    for(int i = 0; i < matrix.Height(); i++)
      for(int j = 0; j < matrix.Width(); j++)
        {
          mp_limb_t r;
          slong a = (slong)matrix.Get(i, j);
          mp_limb_t b = (a < 0) ? a + shift : a;
          NMOD_RED(r, b, mod);
          result.Set(i, j, r);
        }

    return result;
  }

  using El_mat_mul_t
    = std::function<El::Matrix<double>(El::Matrix<double>, El::Matrix<double>)>;
  std::vector<El::Matrix<uint32_t>>
  fmpz_mat_mul_residues(const std::vector<mp_limb_t> &primes,
                        const fmpz_mat_t x, const fmpz_mat_t y,
                        const El_mat_mul_t &El_mat_mul)
  {
    auto xs = fmpz_mat_residues(primes, x);
    auto ys = fmpz_mat_residues(primes, y);
    std::vector<El::Matrix<uint32_t>> result(primes.size());
    for(size_t i = 0; i < primes.size(); ++i)
      {
        result[i] = to_uint32_matrix(primes[i], El_mat_mul(xs[i], ys[i]));
      }
    return result;
  }

  // In SDPB,
  void from_residues(fmpz_mat_t output,
                     const std::vector<El::Matrix<uint32_t>> &residues,
                     const std::vector<mp_limb_t> &primes, int sign)
  {
    slong num_primes = primes.size();
    //    slong n = arg->n;
    //    slong Cstartrow = arg->Cstartrow;
    //    slong Cstoprow = arg->Cstoprow;
    //    uint32_t * bigC = arg->bigC;
    //    fmpz ** Crows = arg->Crows;

    fmpz_comb_t comb;
    fmpz_comb_init(comb, primes.data(), num_primes);
    fmpz_comb_temp_t comb_temp;
    mp_limb_t *r;

    //    int sign = arg->sign;

    fmpz_comb_temp_init(comb_temp, comb);
    r = FLINT_ARRAY_ALLOC(num_primes, mp_limb_t);

    for(slong i = 0; i < output->r; i++)
      {
        for(slong j = 0; j < output->c; j++)
          {
            for(slong k = 0; k < num_primes; k++)
              r[k] = residues[k].Get(i, j);

            fmpz_multi_CRT_ui(&output->rows[i][j], r, comb, comb_temp, sign);
          }
      }

    flint_free(r);
    fmpz_comb_temp_clear(comb_temp);
  }

  void
  fmpz_mat_mul_El_double(fmpz_mat_t output, const fmpz_mat_t x,
                         const fmpz_mat_t y, const El_mat_mul_t &El_mat_mul)
  {
    slong Abits = fmpz_mat_max_bits(x);
    slong Bbits = fmpz_mat_max_bits(y);
    flint_bitcnt_t Cbits;
    int sign = 0;

    if(Abits < 0)
      {
        sign = 1;
        Abits = -Abits;
      }

    if(Bbits < 0)
      {
        sign = 1;
        Bbits = -Bbits;
      }

    Cbits = Abits + Bbits + FLINT_BIT_COUNT(x->c);

    auto primes = calculate_primes(Cbits + sign, x->c);
    auto residues = fmpz_mat_mul_residues(primes, x, y, El_mat_mul);
    from_residues(output, residues, primes, sign);
  }
  void fmpz_mat_mul_El_double_noblas(fmpz_mat_t output, const fmpz_mat_t x,
                                     const fmpz_mat_t y)
  {
    fmpz_mat_mul_El_double(output, x, y, El_multiply_noblas);
  }
  void fmpz_mat_mul_El_double_blas(fmpz_mat_t output, const fmpz_mat_t x,
                                   const fmpz_mat_t y)
  {
    fmpz_mat_mul_El_double(output, x, y, El_multiply_blas);
  }
}

namespace
{
  void fmpz_t_to_BigFloat(const fmpz_t input, El::BigFloat &output)
  {
    fmpz_get_mpf(output.gmp_float.get_mpf_t(), input);
  }
  void BigFloat_to_fmpz_t(const El::BigFloat &input, fmpz_t output)
  {
    fmpz_set_mpf(output, input.gmp_float.get_mpf_t());
  }

  // normalize columns
  void normalize_columns(const El::Matrix<El::BigFloat> &matrix,
                         El::Matrix<El::BigFloat> &output,
                         std::vector<El::BigFloat> &column_norms)
  {
    column_norms.resize(matrix.Width(), 0);
    for(int i = 0; i < matrix.Height(); ++i)
      for(int j = 0; j < matrix.Width(); ++j)
        {
          column_norms[j] += matrix(i, j) * matrix(i, j);
        }

    for(auto &norm : column_norms)
      {
        norm = Sqrt(norm);
      }

    output.Resize(matrix.Height(), matrix.Width());
    for(int i = 0; i < matrix.Height(); ++i)
      for(int j = 0; j < matrix.Width(); ++j)
        {
          output(i, j) = matrix(i, j) / column_norms[j];
        }
  }

  void normalize_rows(const El::Matrix<El::BigFloat> &matrix,
                      El::Matrix<El::BigFloat> &output,
                      std::vector<El::BigFloat> &row_norms)
  {
    row_norms.resize(matrix.Height(), 0);
    for(int i = 0; i < matrix.Height(); ++i)
      for(int j = 0; j < matrix.Width(); ++j)
        {
          row_norms[i] += matrix(i, j) * matrix(i, j);
        }

    for(auto &norm : row_norms)
      {
        norm = Sqrt(norm);
      }

    output.Resize(matrix.Height(), matrix.Width());
    for(int i = 0; i < matrix.Height(); ++i)
      for(int j = 0; j < matrix.Width(); ++j)
        {
          output(i, j) = matrix(i, j) / row_norms[i];
        }
  }

  // multiply by 2^bit_shift
  // convert to big integers
  // NB: normalize matrix before calling this!
  void to_Fmpz_Matrix(const El::Matrix<El::BigFloat> &matrix,
                      Fmpz_Matrix &output, int bit_shift)
  {
    output.Resize(matrix.Height(), matrix.Width());
    for(int i = 0; i < matrix.Height(); ++i)
      for(int j = 0; j < matrix.Width(); ++j)
        {
          BigFloat_to_fmpz_t(matrix(i, j) << bit_shift, output(i, j));
        }
  }

  // divide by 2^bit_shift
  // convert to BigFloat
  void from_Fmpz_Matrix(const Fmpz_Matrix &int_matrix,
                        El::Matrix<El::BigFloat> &output, int bit_shift)
  {
    output.Resize(int_matrix.Height(), int_matrix.Width());
    for(int i = 0; i < int_matrix.Height(); ++i)
      for(int j = 0; j < int_matrix.Width(); ++j)
        {
          El::BigFloat f;
          fmpz_t_to_BigFloat(int_matrix(i, j), f);
          output(i, j) = f >> bit_shift;
        }
  }

  // NB: call for normalized matrices, otherwise precision will be lost
  void mat_mul_BigFloat_fmpz(const El::Matrix<El::BigFloat> &x,
                             const El::Matrix<El::BigFloat> &y,
                             El::Matrix<El::BigFloat> &output)
  {
    Fmpz_Matrix x_int, y_int;
    int bit_shift = El::gmp::Precision();
    to_Fmpz_Matrix(x, x_int, bit_shift);
    to_Fmpz_Matrix(y, y_int, bit_shift);
    Fmpz_Matrix result_int(x.Height(), y.Width());
//    fmpz_mat_mul_blas(result_int.fmpz_matrix, x_int.fmpz_matrix, y_int.fmpz_matrix);
    Primes::fmpz_mat_mul_El_double_blas(result_int.fmpz_matrix,
                                        x_int.fmpz_matrix, y_int.fmpz_matrix);
    // divide by 2^(2*bit_shift) because each of two int matrices should be
    // divided by 2^bit_shift
    from_Fmpz_Matrix(result_int, output, 2 * bit_shift);
  }

  void mat_mul_BigFloat_El(const El::Matrix<El::BigFloat> &x,
                           const El::Matrix<El::BigFloat> &y,
                           El::Matrix<El::BigFloat> &output)
  {
    El::Gemm<El::BigFloat>(El::OrientationNS::NORMAL,
                           El::OrientationNS::NORMAL, 1, x, y, output);
  }
}

// helper functions
namespace
{
  void diff(const fmpz_mat_t a, const fmpz_mat_t b)
  {
    INFO("diff fmpz_mat_t");
    REQUIRE(fmpz_mat_equal(a, b) != 0);
  }

  void diff(const Fmpz_Matrix &a, const Fmpz_Matrix &b)
  {
    DIFF(a.fmpz_matrix, b.fmpz_matrix);
  }

  void
  fmpz_mat_mul_naive(fmpz_mat_t result, const fmpz_mat_t x, const fmpz_mat_t y)
  {
    fmpz_mat_zero(result);
    for(mp_limb_signed_t row = 0; row < x->r; ++row)
      for(mp_limb_signed_t col = 0; col < y->c; ++col)
        {
          auto element = fmpz_mat_entry(result, row, col);
          for(mp_limb_signed_t i = 0; i < x->c; ++i)
            {
              fmpz_addmul(element, fmpz_mat_entry(x, row, i),
                          fmpz_mat_entry(y, i, col));
            }
        }
  }

  Fmpz_Matrix
  random_fmpz_matrix(int height, int width, int bits, flint_rand_t rand)
  {
    Fmpz_Matrix result(height, width);
    fmpz_mat_randbits(result.fmpz_matrix, rand, bits);
    return result;
  }

  using mat_mul_t
    = std::function<void(fmpz_mat_t, const fmpz_mat_t, const fmpz_mat_t)>;

  void Fmpz_Matrix_mul(Fmpz_Matrix &output, const Fmpz_Matrix &x,
                       const Fmpz_Matrix &y, const mat_mul_t &mat_mul_impl)
  {
    output.Resize(x.Height(), y.Width());
    mat_mul_impl(output.fmpz_matrix, x.fmpz_matrix, y.fmpz_matrix);
  }
}

TEST_CASE("mat_mul_BigFloat")
{
  if(El::mpi::Rank() != 0)
    return;
  El::InitializeRandom(true);

  int m = 10, n = 20, k = 1000;

  El::Matrix<El::BigFloat> x = Test_Util::random_matrix(m, k);
  El::Matrix<El::BigFloat> y = Test_Util::random_matrix(k, n);

  // Normalize rows of x and columns of y
  std::vector<El::BigFloat> x_row_norms, y_col_norms;
  normalize_rows(x, x, x_row_norms);
  normalize_columns(y, y, y_col_norms);

  auto eps = El::BigFloat(1) >> El::gmp::Precision();
  REQUIRE(eps < 1.0);

  SECTION("normalize")
  {
    //    DIFF(Norm(x), Sqrt(El::BigFloat(x.Width())));
    //    DIFF(Norm(y), Sqrt(El::BigFloat(y.Height())));

    // norm for fisrt row
    El::BigFloat norm_squared_x0 = 0;
    // norm for first column
    El::BigFloat norm_squared_y0 = 0;
    for(int i = 0; i < x.Width(); ++i)
      {
        norm_squared_x0 += x(0, i) * x(0, i);
        norm_squared_y0 += y(i, 0) * y(i, 0);
      }
    //    norm_sq_x0 = Sqrt(norm_sq_x0);
    //    norm_sq_y0 = Sqrt(norm_sq_y0);
    DIFF(norm_squared_x0, 1.0);
    DIFF(norm_squared_y0, 1.0);

    //    El::Matrix<El::BigFloat> x2, y_2;
    //    std::vector<El::BigFloat> x_col_norms_2, y_row_norms_2;
    //    normalize_columns(x, x2, x_col_norms_2);
    //    normalize_rows(y, y_2, y_row_norms_2);
    //    DIFF(x, x2);
    //    DIFF(y, y_2);
  }

  SECTION("to and from fmpz")
  {
    // sanity check for bit shift
    REQUIRE((El::BigFloat(1) << 1) == El::BigFloat(2));

    for(double d : std::vector<double>{1e-10, 0.1, 1, 10, 1e10})
      {
        DYNAMIC_SECTION(d)
        {
          // input
          auto f = El::BigFloat(d) / 3.0;
          // output
          El::BigFloat f_out;

          int bits = El::gmp::Precision();

          {
            fmpz_t big_int;
            BigFloat_to_fmpz_t(f << bits, big_int);
            fmpz_t_to_BigFloat(big_int, f_out);
          }

          REQUIRE(Round(f_out) == f_out);
          f_out >>= bits;

          auto f_shifted = f << bits;
          auto f2_shifted = f_out << bits;
          CAPTURE(Round(f_shifted));
          CAPTURE(Round(f2_shifted));

          DIFF(Round(f_shifted), Round(f2_shifted));
          CAPTURE(f_shifted - Round(f_shifted));
          CAPTURE(f2_shifted - Round(f2_shifted));

          CAPTURE(eps);

          CAPTURE(f);
          CAPTURE(f_out);
          CAPTURE(f - f_out);
          REQUIRE(Abs(f - f_out) < eps);
          // TODO calling fmpz_clear somehow breaks other sections!
          // fmpz_clear(big_int);
        }
      }
  }

  SECTION("to and from Fmpz_Matrix")
  {
    El::Matrix<El::BigFloat> x_out;
    int bits = El::gmp::Precision();

    {
      Fmpz_Matrix x_int;
      to_Fmpz_Matrix(x, x_int, bits);
      from_Fmpz_Matrix(x_int, x_out, bits);
    }
    {
      for(int i = 0; i < x.Height(); ++i)
        for(int j = 0; j < x.Width(); ++j)
          {
            CAPTURE(x(i, j));
            CAPTURE(x_out(i, j));
            REQUIRE(Abs(x(i, j) - x_out(i, j)) < eps);
          }
      //
      //      double size = dx.Height() * dx.Width();
      //      CAPTURE(dx(0, 0));
      //      REQUIRE(El::Norm(dx) < eps * sqrt(size));
    }
  }

  // TODO check correctness by increasing precision
  SECTION("mat_mul")
  {
    El::BigFloat res00 = 0;
    El::BigFloat norm_x0 = 0;
    El::BigFloat norm_y0 = 0;
    for(int i = 0; i < x.Width(); ++i)
      {
        res00 += x(0, i) * y(i, 0);
        norm_x0 += x(0, i) * x(0, i);
        norm_y0 += y(i, 0) * y(i, 0);
      }
    norm_x0 = Sqrt(norm_x0);
    norm_y0 = Sqrt(norm_y0);
    CAPTURE(res00);
    CAPTURE(norm_x0);
    CAPTURE(norm_y0);

    El::Matrix<El::BigFloat> result_El, result_fmpz;
    mat_mul_BigFloat_El(x, y, result_El);
    mat_mul_BigFloat_fmpz(x, y, result_fmpz);

    REQUIRE(Abs(result_El(0, 0) - res00) < 2 * eps * El::Sqrt(x.Width()));
    REQUIRE(Abs(result_fmpz(0, 0) - res00) < 2 * eps * El::Sqrt(x.Width()));

    int height = result_El.Height();
    int width = result_El.Width();
    El::BigFloat size = height * width;

    for(int i = 0; i < height; ++i)
      for(int j = 0; j < width; ++j)
        {
          CAPTURE(i);
          CAPTURE(j);
          const auto &elt_El = result_El(i, j);
          const auto &elt_fmpz = result_fmpz(i, j);
          // normalization
          REQUIRE(Abs(elt_El) <= 1.0);
          REQUIRE(Abs(elt_fmpz) <= 1.0);

          // Sum of x.Width() terms =>
          REQUIRE(Abs(elt_El - elt_fmpz) < 2 * eps * El::Sqrt(x.Width()));
        }

    auto delta = result_fmpz;
    delta -= result_El;

    REQUIRE(El::Norm(delta) < 2 * eps * Sqrt(size));
  }
}

TEST_CASE("benchmark mat_mul_BigFloat", "[!benchmark]")
{
  if(El::mpi::Rank() != 0)
    return;
  El::InitializeRandom(true);

  int m = GENERATE(10, 100, 500, 1000);
  int n = m;
  int k = m;

  El::Matrix<El::BigFloat> x = Test_Util::random_matrix(m, k);
  El::Matrix<El::BigFloat> y = Test_Util::random_matrix(k, n);

  // Normalize rows of x and columns of y
  {
    std::vector<El::BigFloat> x_row_norms, y_col_norms;
    normalize_rows(x, x, x_row_norms);
    normalize_columns(y, y, y_col_norms);
  }

  DYNAMIC_SECTION("n=" << n << ", m=" << m << ", k=" << k)
  {
    El::Matrix<El::BigFloat> result_El(m, n), result_fmpz(m, n);

    BENCHMARK("mat_mul_BigFloat_fmpz")
    {
      return mat_mul_BigFloat_fmpz(x, y, result_fmpz);
    };
    BENCHMARK("mat_mul_BigFloat_El")
    {
      return mat_mul_BigFloat_El(x, y, result_El);
    };
  }
}

TEST_CASE("bigint_matrix")
{
  if(El::mpi::Rank() != 0)
    return;
  CHECK(flint_get_num_threads() == 1);

  flint_rand_t my_rand_t;
  flint_randinit(my_rand_t);

  int m = 100, n = 110, k = 300;
  int bits = 256;

  Fmpz_Matrix x = random_fmpz_matrix(m, k, bits, my_rand_t);
  Fmpz_Matrix y = random_fmpz_matrix(k, n, bits, my_rand_t);
  //  Fmpz_Matrix result(x.Height(), y.Width());

  // TODO conflicts with El::gmp::precision?
  //  mpf_set_default_prec(bits);

  SECTION("Primes")
  {
    auto primes = Primes::calculate_primes(bits, x.Width());
    CAPTURE(primes);
    //    FAIL("TODO");
  }

  SECTION("Correctness")
  {
    std::vector<mat_mul_t> mat_muls
      = {fmpz_mat_mul, fmpz_mat_mul_blas, fmpz_mat_mul_naive,
         Primes::fmpz_mat_mul_El_double_blas,
         Primes::fmpz_mat_mul_El_double_noblas};

    std::vector<Fmpz_Matrix> results(mat_muls.size());

    for(size_t i = 0; i < mat_muls.size(); ++i)
      {
        Fmpz_Matrix_mul(results.at(i), x, y, mat_muls[i]);
        if(i != 0)
          {
            CAPTURE(i);
            DIFF(results[i], results[i - 1]);
          }
      }
  }
}

TEST_CASE("fmpz multiplication benchmark", "[!benchmark]")
{
  if(El::mpi::Rank() != 0)
    return;
  CHECK(flint_get_num_threads() == 1);

  flint_rand_t my_rand_t;
  flint_randinit(my_rand_t);

  int m = 500, n = 500, k = 500;
  int bits = 840;

  Fmpz_Matrix x = random_fmpz_matrix(m, n, bits, my_rand_t);
  Fmpz_Matrix y = random_fmpz_matrix(n, k, bits, my_rand_t);
  // TODO allocate sufficient number of bits?
  //  Fmpz_Matrix result(x.Height(),y.Width());

#define BENCHMARK_mat_mul(mat_mul_impl)                                       \
  BENCHMARK(#mat_mul_impl)                                                    \
  {                                                                           \
    Fmpz_Matrix result;                                                       \
    return Fmpz_Matrix_mul(result, x, y, mat_mul_impl);                       \
  }

  BENCHMARK_mat_mul(fmpz_mat_mul_blas);
  BENCHMARK_mat_mul(fmpz_mat_mul);
  BENCHMARK_mat_mul(fmpz_mat_mul_multi_mod);
  BENCHMARK_mat_mul(Primes::fmpz_mat_mul_El_double_blas);
  BENCHMARK_mat_mul(Primes::fmpz_mat_mul_El_double_noblas);
  //  BENCHMARK_mat_mul(fmpz_mat_mul_naive);

#undef BENCHMARK_mat_mul
}

El::BigFloat
to_BigFloat(const fmpz_t input, int base, char *str_buffer = nullptr)
{
  // TODO allocate str_buffer of sufficient length once, outside the loop
  //  int base = 2; //TODO which base is the fastest?
  str_buffer = fmpz_get_str(str_buffer, base, input);
  return El::BigFloat(str_buffer, base);
}

El::BigFloat to_BigFloat_fmpz_get_mpf(const fmpz_t input)
{
  El::BigFloat f;
  //  El::Gemm(El::OrientationNS::NORMAL,El::OrientationNS::NORMAL,1,A,B,result);
  fmpz_get_mpf(f.gmp_float.get_mpf_t(), input);
  return f;
}

// Various tests

#if 0

TEST_CASE("to_BigFloat ", "[!benchmark]")
{
  if(El::mpi::Rank() != 0)
    return;

  mp_bitcnt_t bits = 1024;
  mpf_set_default_prec(bits);

  flint_rand_t my_rand_t;
  flint_randinit(my_rand_t);
  fmpz_t integer;
  fmpz_init(integer);
  fmpz_randbits(integer, my_rand_t, bits);

  char str_buffer[bits + 2]; // TODO

  SECTION("test")
  {
    fmpz_print(integer);
    El::Output();
    for(int base : {2, 10, 16, 32, 62})
      {
        auto f = to_BigFloat(integer, base);
        auto f2 = to_BigFloat_fmpz_get_mpf(integer);
        REQUIRE(f == f2);
        REQUIRE(f.Precision() == El::gmp::Precision());
        REQUIRE(f2.Precision() == El::gmp::Precision());
        El::Output(
          to_BigFloat(integer, base)
          - El::BigFloat(
            "-1559258775283944826146106259936745880191007710663592180785545871"
            "62570881239567576804461126115887903799308419845098579180815670096"
            "03552187487090896179963826919606856010375230866981812886067771946"
            "038130439758786259360796835828656706857947976367181795"
            "5144283945749615768573725580291910494735428411976050787788916"));
      }
  }

  //  int base = GENERATE(2, 4, 6, 10, 12, 16, 32, 62);
  //  DYNAMIC_SECTION("base=" << base)
  //  BENCHMARK("to_BigFloat") { return to_BigFloat(integer, base); };

#define BENCHMARK_to_BigFloat(base)                                           \
  BENCHMARK(std::to_string(base))                                             \
  {                                                                           \
    return to_BigFloat(integer, base, str_buffer);                            \
  };

  BENCHMARK_to_BigFloat(2);
  BENCHMARK_to_BigFloat(8);
  BENCHMARK_to_BigFloat(10);
  BENCHMARK_to_BigFloat(16);
  BENCHMARK_to_BigFloat(32);
  BENCHMARK_to_BigFloat(33);
  BENCHMARK_to_BigFloat(62);
#undef BENCHMARK_to_BigFloat
  // This is ~10x faster
  BENCHMARK("fmpz_get_mpf")
  {
    El::BigFloat f;
    // TODO check
    fmpz_get_mpf(f.gmp_float.get_mpf_t(), integer);
    return f;
  };
}

using Boost_Float = boost::multiprecision::mpfr_float;

inline std::string to_string(const Boost_Float &boost_float)
{
  // Using a stringstream seems to the best way to convert between
  // MPFR and GMP.  It may lose a bit or two since string
  // conversion is not sufficient for round-tripping.
  std::stringstream ss;
  set_stream_precision(ss);
  ss << boost_float;
  return ss.str();
}
inline El::BigFloat
to_BigFloat_via_stringstream(const Boost_Float &boost_float)
{
  return El::BigFloat(to_string(boost_float));
}

inline El::BigFloat to_BigFloat_via_str(const Boost_Float &boost_float)
{
  return El::BigFloat(boost_float.str());
}

inline El::BigFloat
to_BigFloat_via_mpfr_get_str(const Boost_Float &boost_float, int base)
{
  mpfr_exp_t exponent;
  char *str = mpfr_get_str(nullptr, &exponent, base, 0,
                           boost_float.backend().data(), MPFR_RNDN);
  El::BigFloat result;
  mpf_set_str(result.gmp_float.get_mpf_t(), str,
              base); // TODO fails, add exponent~
  mpfr_free_str(str);
  FAIL("Please account for exponent!");
  return result;
}

TEST_CASE("Boost_Float to BigFloat", "[!benchmark]")
{
  Boost_Float::default_precision(std::ceil(El::gmp::Precision() * std::log10(2.0)) + 1);
  Boost_Float boost_float = 1.12312e101;
  boost_float /= 0.34533;
  DIFF(to_BigFloat_via_stringstream(boost_float), to_BigFloat_via_str(boost_float));

  BENCHMARK("stringstream")
  {
    return to_BigFloat_via_stringstream(boost_float);
  };
  BENCHMARK("str")
  {
    return to_BigFloat_via_str(boost_float);
  };
}



namespace DSD
{
  // DSD code:

  bool compare_mpz_vectors(const std::vector<mpz_class> &x,
                           const std::vector<mpz_class> &y)
  {
    if(x.size() != y.size())
      {
        return false;
      }
    bool result = true;
    for(uint i = 0; i < x.size(); ++i)
      {
        result = result && (x[i] == y[i]);
      }
    return result;
  }

  void
  copy_fmpz_mat_t_to_mpz_class(fmpz_mat_t x, std::vector<mpz_class> &x_mpz,
                               const slong m, const slong n)
  {
    for(uint i = 0; i < m; ++i)
      {
        for(uint j = 0; j < n; ++j)
          {
            fmpz_get_mpz(x_mpz[i * n + j].get_mpz_t(),
                         fmpz_mat_entry(x, i, j));
          }
      }
  }

  void mpz_mat_multiply_classical(const int m, const int n, const int k,
                                  const std::vector<mpz_class> &x,
                                  const std::vector<mpz_class> &y,
                                  std::vector<mpz_class> &result)
  {
    for(uint i = 0; i < m; ++i)
      {
        for(uint l = 0; l < k; ++l)
          {
            result[k * i + l] = 0;
            for(uint j = 0; j < n; ++j)
              {
                result[k * i + l] += x[n * i + j] * y[k * j + l];
              }
          }
      }
  }

  void flint_fmpz_mat_test(const slong m, const slong n, const slong k,
                           const int num_tests)
  {
    std::cout << "FLINT running with " << flint_get_num_threads()
              << " thread(s)" << std::endl;
    std::cout << "Multipliying matrices of size " << m << "x" << n << " and "
              << n << "x" << k << std::endl;

    fmpz_mat_t x;
    fmpz_mat_t y;
    fmpz_mat_t result;
    fmpz_mat_t result_blas;
    fmpz_mat_init(x, m, n);
    fmpz_mat_init(y, n, k);
    fmpz_mat_init(result, m, k);
    fmpz_mat_init(result_blas, m, k);

    flint_rand_t my_rand_t;
    flint_randinit(my_rand_t);
    fmpz_mat_randbits(x, my_rand_t, 840);
    fmpz_mat_randbits(y, my_rand_t, 840);

    std::vector<mpz_class> x_mpz(m * n);
    std::vector<mpz_class> y_mpz(n * k);
    std::vector<mpz_class> result_mpz(n * k);
    std::vector<mpz_class> result_flint_mpz(n * k);
    std::vector<mpz_class> result_flint_blas_mpz(n * k);
    copy_fmpz_mat_t_to_mpz_class(x, x_mpz, m, n);
    copy_fmpz_mat_t_to_mpz_class(y, y_mpz, n, k);

    const auto start_flint = std::chrono::steady_clock::now();
    for(uint test = 0; test < num_tests; ++test)
      {
        std::cout << "flint test: " << test << std::endl;
        fmpz_mat_mul(result, x, y);
      }
    const auto end_flint = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds_flint
      = end_flint - start_flint;
    copy_fmpz_mat_t_to_mpz_class(result, result_flint_mpz, m, k);

    const auto start_flint_blas = std::chrono::steady_clock::now();
    for(uint test = 0; test < num_tests; ++test)
      {
        std::cout << "flint_blas test: " << test << std::endl;
        fmpz_mat_mul_blas(result_blas, x, y);
      }
    const auto end_flint_blas = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds_flint_blas
      = end_flint_blas - start_flint_blas;
    copy_fmpz_mat_t_to_mpz_class(result_blas, result_flint_blas_mpz, m, k);

    const auto start_classical = std::chrono::steady_clock::now();
    for(uint test = 0; test < num_tests; ++test)
      {
        std::cout << "classical test: " << test << std::endl;
        mpz_mat_multiply_classical(m, n, k, x_mpz, y_mpz, result_mpz);
      }
    const auto end_classical = std::chrono::steady_clock::now();
    const std::chrono::duration<double> elapsed_seconds_classical
      = end_classical - start_classical;

    CHECK(compare_mpz_vectors(result_flint_mpz, result_mpz));
    std::cout << "result_flint_mpz      == result_mpz: "
              << (compare_mpz_vectors(result_flint_mpz, result_mpz) ? "True"
                                                                    : "False")
              << std::endl;
    CHECK(compare_mpz_vectors(result_flint_blas_mpz, result_mpz));
    std::cout << "result_flint_blas_mpz == result_mpz: "
              << (compare_mpz_vectors(result_flint_blas_mpz, result_mpz)
                    ? "True"
                    : "False")
              << std::endl;

    std::cout << "flint     : " << elapsed_seconds_flint.count() << std::endl;
    std::cout << "flint_blas: " << elapsed_seconds_flint_blas.count()
              << std::endl;
    std::cout << "classical : " << elapsed_seconds_classical.count()
              << std::endl;
    std::cout << "classical/flint     : "
              << elapsed_seconds_classical.count()
                   / elapsed_seconds_flint.count()
              << std::endl;
    std::cout << "classical/flint_blas: "
              << elapsed_seconds_classical.count()
                   / elapsed_seconds_flint_blas.count()
              << std::endl;
  }
}

// TEST_CASE("DSD")
//{
//   FAIL();
//   int num_tests = 1;
//   DSD::flint_fmpz_mat_test(500, 500, 500, num_tests);
// }

#endif