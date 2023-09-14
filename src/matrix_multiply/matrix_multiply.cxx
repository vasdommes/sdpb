#include <cblas.h>
#include <flint/flint.h>
#include <flint/fmpz.h>
#include <flint/nmod.h>

#include <El.hpp>

#include "Comb.hxx"

#include "matrix_multiply.hxx"
#include "fmpz_util.hxx"

// code adopted from flint mul_blas.c
namespace
{

#define MAX_BLAS_DP_INT (UWORD(1) << 53)

  // see mul_blas.c, _tod_worker()
  double uint32_t_residue_to_double(uint32_t value, nmod_t mod)
  {
    // return (int32_t)(value - (mod.n & (-(uint32_t)((int32_t)(mod.n/2 - value) < 0))));

    // This is equivalent to the above formula, but more readable:
    if(mod.n / 2 < value)
      return (int32_t)(value - (mod.n & UINT32_MAX));
    else
      return value;
  }

  // see mul_blas.c, _fromd_worker()
  uint32_t double_to_uint32_t_residue(double value, nmod_t mod)
  {
    ulong shift = ((2 * MAX_BLAS_DP_INT) / mod.n) * mod.n;
    mp_limb_t r;
    slong a = (slong)value;
    mp_limb_t b = (a < 0) ? a + shift : a;
    NMOD_RED(r, b, mod); // r := b % mod.n
    return (uint32_t)r;
  }

  double _reduce_uint32(mp_limb_t a, nmod_t mod)
  {
    mp_limb_t r;
    NMOD_RED(r, a, mod);
    return (uint32_t)r;
  }

  double _reduce_double(mp_limb_t a, nmod_t mod)
  {
    return uint32_t_residue_to_double(_reduce_uint32(a, mod), mod);
  }

  void
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
                out[l * stride] = _reduce_double(t, lu[i].mod0);
                l++;
                out[l * stride] = _reduce_double(t, lu[i].mod1);
                l++;
                out[l * stride] = _reduce_double(t, lu[i].mod2);
                l++;
              }
            else if(lu[i].mod1.n != 0)
              {
                FLINT_ASSERT(l + 2 <= C->num_primes);
                out[l * stride] = _reduce_double(t, lu[i].mod0);
                l++;
                out[l * stride] = _reduce_double(t, lu[i].mod1);
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
}

// Step 4: compute residues and put them to shared window
// NB: input_block is normalized matrix, multiplied by 2^N
void compute_matrix_residues(
  Block_Residue_Matrices_Window<double> &output_window,
  const El::DistMatrix<El::BigFloat> &input_block, Comb &comb,
  size_t block_index_in_node)
{
  assert(output_window.width == input_block.Width());

  // for each input_matrix element
  // for each prime_index=0..primes.size():
  // - Calculate input_matrix(i,j) mod primes[prime_index]
  // - Write the result to output_window, with
  //   offset = start_offset + prime_index * prime_stride
  fmpz_t bigint_value;
  fmpz_init(bigint_value);
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
                                         bigint_value, comb.comb,
                                         comb.comb_temp);
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
                   const Residue_Matrices_Window<double> &window, Comb &comb,
                   size_t i, size_t j)
{
  int sign = 1; // means that negative values are allowed
  size_t num_primes = comb.num_primes;

  std::vector<mp_limb_t> residues(num_primes);
  for(size_t prime_index = 0; prime_index < num_primes; prime_index++)
    {
      double d = window.residues.at(prime_index)(i, j);
      assert(d > (double)std::numeric_limits<slong>::min());
      assert(d < (double)std::numeric_limits<slong>::max());
      auto &mod = comb.mods.at(prime_index);
      residues.at(prime_index) = double_to_uint32_t_residue(d, mod);
    }

  fmpz_multi_CRT_ui(output, residues.data(), comb.comb, comb.comb_temp, sign);
}

void calculate_Block_Matrix_square(
  El::DistMatrix<El::BigFloat> &output,
  // Blocks stored for a given rank
  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
  // Indices of input_normalized_blocks in blocks_window
  const std::vector<size_t> &block_indices_for_window,
  Block_Residue_Matrices_Window<double> &blocks_window,
  Residue_Matrices_Window<double> &result_window, Comb &comb)
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
      compute_matrix_residues(blocks_window, block, comb, block_index);
    }
  // wait for all ranks to fill blocks_window
  blocks_window.Fence();

  // Square each residue matrix
  auto comm = blocks_window.Comm();
  for(size_t prime_index = 0; prime_index < comb.num_primes; ++prime_index)
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
  fmpz_init(big_int_value);
  El::BigFloat big_float_value;
  for(int i = 0; i < output.LocalHeight(); ++i)
    for(int j = 0; j < output.LocalWidth(); ++j)
      {
        int global_i = output.GlobalRow(i);
        int global_j = output.GlobalCol(j);
        from_residues(big_int_value, result_window, comb, global_i, global_j);
        fmpz_t_to_BigFloat(big_int_value, big_float_value);
        output.SetLocal(i, j, big_float_value);
      }

  fmpz_clear(big_int_value); // TODO use RAII wrapper?
}