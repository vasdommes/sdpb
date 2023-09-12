#include <cblas.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/fmpz_mat.h>
#include <flint/ulong_extras.h>
#include <flint/nmod.h>

#include <El.hpp>

#include "Fmpz_Matrix.hxx"
#include "Primes.hxx"

#include "matrix_multiply.hxx"
#include "fmpz_util.hxx"

// code adopted from flint mul_blas.c
namespace
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

  static void
  fmpz_multi_mod_uint32_stride(double *out, slong stride, const fmpz_t input,
                               const fmpz_comb_t C, fmpz_comb_temp_t CT)
  {
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
  }

  El::Matrix<double>
  El_multiply_noblas(const El::Matrix<double> &x, const El::Matrix<double> &y)
  {
    El::Matrix<double> result;
    El::Gemm(El::OrientationNS::NORMAL, El::OrientationNS::NORMAL, 1.0, x, y,
             result);
    return result;
  }

  void
  El_multiply_blas(El::Matrix<double> &result, const El::Matrix<double> &x,
                   const El::Matrix<double> &y)
  {
    int M = x.Height();
    int N = y.Width();
    int K = x.Width();
    result.Resize(M, N);

    const double alpha = 1.0;
    const double beta = 0.0;
    const double *A = x.LockedBuffer();
    const double *B = y.LockedBuffer();
    double *C = result.Buffer();
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

  //  using El_mat_mul_t
  //    = std::function<El::Matrix<double>(El::Matrix<double>,
  //    El::Matrix<double>)>;
  //  std::vector<El::Matrix<uint32_t>>
  //  fmpz_mat_mul_residues(const std::vector<mp_limb_t> &primes,
  //                        const fmpz_mat_t x, const fmpz_mat_t y,
  //                        const El_mat_mul_t &El_mat_mul)
  //  {
  //    auto xs = fmpz_mat_residues(primes, x);
  //    auto ys = fmpz_mat_residues(primes, y);
  //    std::vector<El::Matrix<uint32_t>> result(primes.size());
  //    for(size_t i = 0; i < primes.size(); ++i)
  //      {
  //        result[i] = to_uint32_matrix(primes[i], El_mat_mul(xs[i], ys[i]));
  //      }
  //    return result;
  //  }

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

  //  void
  //  fmpz_mat_mul_El_double(fmpz_mat_t output, const fmpz_mat_t x,
  //                         const fmpz_mat_t y, const El_mat_mul_t &El_mat_mul)
  //  {
  //    slong Abits = fmpz_mat_max_bits(x);
  //    slong Bbits = fmpz_mat_max_bits(y);
  //    flint_bitcnt_t Cbits;
  //    int sign = 0;
  //
  //    if(Abits < 0)
  //      {
  //        sign = 1;
  //        Abits = -Abits;
  //      }
  //
  //    if(Bbits < 0)
  //      {
  //        sign = 1;
  //        Bbits = -Bbits;
  //      }
  //
  //    Cbits = Abits + Bbits + FLINT_BIT_COUNT(x->c);
  //
  //    auto primes = calculate_primes(Cbits + sign, x->c);
  //    auto residues = fmpz_mat_mul_residues(primes, x, y, El_mat_mul);
  //    from_residues(output, residues, primes, sign);
  //  }
  //  void fmpz_mat_mul_El_double_noblas(fmpz_mat_t output, const fmpz_mat_t x,
  //                                     const fmpz_mat_t y)
  //  {
  //    fmpz_mat_mul_El_double(output, x, y, El_multiply_noblas);
  //  }
  //  void fmpz_mat_mul_El_double_blas(fmpz_mat_t output, const fmpz_mat_t x,
  //                                   const fmpz_mat_t y)
  //  {
  //    fmpz_mat_mul_El_double(output, x, y, El_multiply_blas);
  //  }
}

// Step 4: compute residues and put them to shared window
// NB: input_block is normalized matrix, multiplied by 2^N
void compute_matrix_residues(
  Block_Residue_Matrices_Window<double> &output_window,
  const El::DistMatrix<El::BigFloat> &input_block, Primes &primes,
  size_t block_index_in_node)
{
  assert(output_window.width == input_block.Width());

  // for each input_matrix element
  // for each prime_index=0..primes.size():
  // - Calculate input_matrix(i,j) mod primes[prime_index]
  // - Write the result to output_window, with
  //   offset = start_offset + prime_index * prime_stride
  fmpz_t bigint_value;
  for(int i = 0; i < input_block.Height(); ++i)
    for(int j = 0; j < input_block.Width(); ++j)
      {
        if(input_block.IsLocal(i, j))
          {
            // pointer to the first residue
            double *data = output_window.block_residues.at(0)
                             .at(block_index_in_node)
                             .Buffer(i, j);
            BigFloat_to_fmpz_t(input_block.Get(i, j), bigint_value);
            fmpz_multi_mod_uint32_stride(data, output_window.prime_stride,
                                         bigint_value, primes.comb,
                                         primes.comb_temp);
          }
      }
  fmpz_clear(bigint_value); // TODO wrap with RAII
}

// TODO code style: put output always first or always last?
// output = input^T * input
void calculate_matrix_square(const El::Matrix<double> &input,
                             El::Matrix<double> &output)
{
  // input: KxN matrix
  // output = input^T * input: NxN matrix
  assert(input.Width() == output.Width());
  assert(output.Height() == output.Width());

  CBLAS_LAYOUT layout = CblasColMajor;
  CBLAS_UPLO Uplo = CblasUpper;
  CBLAS_TRANSPOSE Trans = CblasTrans;
  const CBLAS_INDEX N = input.Width();
  const CBLAS_INDEX K = input.Height();
  const double alpha = 1.0;
  const double *A = input.LockedBuffer();
  const CBLAS_INDEX lda = input.LDim();
  const double beta = 0.0;
  double *C = output.Buffer();
  const CBLAS_INDEX ldc = output.LDim();
  // C := alpha * A^T * A + beta * C = (in our case) A^T * A
  cblas_dsyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);

  // TODO use SymmetricMatrix instead of initializing lower half explicitly?
  for(size_t i = 0; i < N; ++i)
    for(size_t j = 0; j < i; ++j)
      output(i, j) = output(j, i);
}

void calculate_matrix_residue_square(
  Residue_Matrices_Window<double> &output_window,
  Block_Residue_Matrices_Window<double> &input_window, size_t prime_index)
{
  const auto &input_matrix = input_window.residues.at(prime_index);
  auto &output_matrix = output_window.residues.at(prime_index);
  calculate_matrix_square(input_matrix, output_matrix);
}

void from_residues(fmpz_t &output,
                   const Residue_Matrices_Window<double> &window,
                   Primes &primes, size_t i, size_t j)
{
  int sign = 1; // means that negative values are allowed
  size_t num_primes = primes.size();

  std::vector<mp_limb_t> residues(num_primes);
  for(size_t prime_index = 0; prime_index < num_primes; prime_index++)
    residues.at(prime_index) = window.residues.at(prime_index)(i, j);

  fmpz_multi_CRT_ui(output, residues.data(), primes.comb, primes.comb_temp,
                    sign);
}

void calculate_Block_Matrix_square(
  El::DistMatrix<El::BigFloat> &output,
  // Blocks stored for a given rank
  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
  // Indices of input_normalized_blocks in blocks_window
  const std::vector<size_t> &block_indices_for_window,
  Block_Residue_Matrices_Window<double> &blocks_window,
  Residue_Matrices_Window<double> &result_window, Primes &primes)
{
  size_t width = output.Width();

  assert(input_normalized_blocks.size() == block_indices_for_window.size());

  assert(output.Height() == output.Width());
  assert(output.Height() == width);
  assert(blocks_window.width == width);

  assert(El::mpi::Congruent(result_window.Comm(), blocks_window.Comm()));
  //  assert(output.DistComm() != El::mpi::COMM_NULL);
  // TODO:
  //  assert(El::mpi::Congruent(output.DistComm(), blocks_window.Comm()));

  // Compute residues
  for(size_t i = 0; i < input_normalized_blocks.size(); ++i)
    {
      // NB: block_indices should enumerate all blocks
      // from all ranks in current node
      size_t block_index = block_indices_for_window.at(i);
      const auto &block = input_normalized_blocks.at(i);
      assert(block.Width() == width);
      compute_matrix_residues(blocks_window, block, primes, block_index);
    }
  // wait for all ranks to fill blocks_window
  blocks_window.Fence();

  // Square each residue matrix
  auto comm = blocks_window.Comm();
  for(size_t prime_index = 0; prime_index < primes.size(); ++prime_index)
    {
      if(prime_index % El::mpi::Size(comm) == El::mpi::Rank(comm))
        {
          const auto &input_matrix = blocks_window.residues.at(prime_index);
          auto &output_matrix = result_window.residues.at(prime_index);
          calculate_matrix_square(input_matrix, output_matrix);
        }
    }
  result_window.Fence();

  fmpz_t big_int_value;
  El::BigFloat big_float_value;
  for(int i = 0; i < output.LocalHeight(); ++i)
    for(int j = 0; j < output.LocalWidth(); ++j)
      {
        int global_i = output.GlobalRow(i);
        int global_j = output.GlobalCol(j);
        from_residues(big_int_value, result_window, primes, global_i,
                      global_j);
        fmpz_t_to_BigFloat(big_int_value, big_float_value);
        output.SetLocal(i, j, big_float_value);
      }

  fmpz_clear(big_int_value); // TODO use RAII wrapper?
}