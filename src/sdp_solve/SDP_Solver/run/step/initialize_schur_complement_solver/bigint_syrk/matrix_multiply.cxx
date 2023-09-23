//#include <cblas.h>
//#include <flint/flint.h>
//#include <flint/fmpz.h>
//#include <flint/nmod.h>
//
//#include <El.hpp>
//
//#include "Fmpz_Comb.hxx"
//#include "fmpz_util.hxx"
//
//#include "matrix_multiply.hxx"
//
//// code adopted from flint mul_blas.c
//namespace
//{
//
//#define MAX_BLAS_DP_INT (UWORD(1) << 53)
//
//  // see mul_blas.c, _tod_worker()
//  // n is the prime
//  // input value is in the range [0;n)
//  // output is in the range (-n/2; n/2]
//  double uint32_t_residue_to_double(uint32_t value, mp_limb_t n)
//  {
//    // return (int32_t)(value - (n & (-(uint32_t)((int32_t)(n/2 - value) < 0))));
//
//    // This is equivalent to the above formula, but more readable:
//    if(n / 2 < value)
//      return (int32_t)(value - (n & UINT32_MAX));
//    else
//      return value;
//  }
//
//  // see mul_blas.c, _fromd_worker()
//  uint32_t double_to_uint32_t_residue(double value, nmod_t mod)
//  {
//    ulong shift = ((2 * MAX_BLAS_DP_INT) / mod.n) * mod.n;
//    mp_limb_t r;
//    slong a = (slong)value;
//    mp_limb_t b = (a < 0) ? a + shift : a;
//    NMOD_RED(r, b, mod); // r := b % mod.n
//    return (uint32_t)r;
//  }
//
//  double _reduce_uint32(mp_limb_t a, nmod_t mod)
//  {
//    mp_limb_t r;
//    NMOD_RED(r, a, mod);
//    return (uint32_t)r;
//  }
//
//  double _reduce_double(mp_limb_t a, nmod_t mod)
//  {
//    return uint32_t_residue_to_double(_reduce_uint32(a, mod), mod.n);
//  }
//
//  void fmpz_multi_mod_uint32_stride(double *out, slong stride,
//                                    const fmpz_t &input, const Fmpz_Comb &comb)
//  {
//    const fmpz_comb_t &C = comb.comb;
//    const fmpz_comb_temp_t &CT = comb.comb_temp;
//
//    slong i, k, l;
//    fmpz *A = CT->A;
//    mod_lut_entry *lu;
//    slong *offsets;
//    slong klen = C->mod_klen;
//    fmpz_t ttt;
//
//    /* high level split */
//    if(klen == 1)
//      {
//        *ttt = A[0];
//        A[0] = *input;
//      }
//    else
//      {
//        _fmpz_multi_mod_precomp(A, C->mod_P, input, -1, CT->T);
//      }
//
//    offsets = C->mod_offsets;
//    lu = C->mod_lu;
//
//    for(k = 0, i = 0, l = 0; k < klen; k++)
//      {
//        slong j = offsets[k];
//
//        for(; i < j; i++)
//          {
//            /* mid level split: depends on FMPZ_MOD_UI_CUTOFF */
//            mp_limb_t t = fmpz_get_nmod(A + k, lu[i].mod);
//
//            /* low level split: 1, 2, or 3 small primes */
//            if(lu[i].mod2.n != 0)
//              {
//                FLINT_ASSERT(l + 3 <= C->num_primes);
//                out[l * stride] = _reduce_double(t, lu[i].mod0);
//                l++;
//                out[l * stride] = _reduce_double(t, lu[i].mod1);
//                l++;
//                out[l * stride] = _reduce_double(t, lu[i].mod2);
//                l++;
//              }
//            else if(lu[i].mod1.n != 0)
//              {
//                assert(l + 2 <= C->num_primes);
//                out[l * stride] = _reduce_double(t, lu[i].mod0);
//                l++;
//                out[l * stride] = _reduce_double(t, lu[i].mod1);
//                l++;
//              }
//            else
//              {
//                assert(l + 1 <= C->num_primes);
//                out[l * stride] = uint32_t_residue_to_double(
//                  (uint32_t)(t), comb.primes.at(l));
//                l++;
//              }
//          }
//      }
//
//    assert(l == C->num_primes);
//
//    if(klen == 1)
//      A[0] = *ttt;
//  }
//
//  // Step 4: compute residues and put them to shared window
//  // NB: input_block is normalized matrix, multiplied by 2^N
//  void compute_matrix_residues(
//    size_t block_index_in_node,
//    const El::DistMatrix<El::BigFloat> &input_block, Fmpz_Comb &comb,
//    Block_Residue_Matrices_Window<double> &block_residues_window)
//  {
//    assert(block_residues_window.width == input_block.Width());
//
//    // for each input_matrix element
//    // for each prime_index=0..primes.size():
//    // - Calculate input_matrix(i,j) mod primes[prime_index]
//    // - Write the result to output_window, with
//    //   offset = start_offset + prime_index * prime_stride
//    fmpz_t bigint_value;
//    fmpz_init(bigint_value);
//    for(int i = 0; i < input_block.Height(); ++i)
//      for(int j = 0; j < input_block.Width(); ++j)
//        {
//          if(input_block.IsLocal(i, j))
//            {
//              // pointer to the first residue
//              double *data = block_residues_window.block_residues.at(0)
//                               .at(block_index_in_node)
//                               .Buffer(i, j);
//              BigFloat_to_fmpz_t(input_block.Get(i, j), bigint_value);
//              fmpz_multi_mod_uint32_stride(
//                data, block_residues_window.prime_stride, bigint_value, comb);
//            }
//        }
//    fmpz_clear(bigint_value); // TODO wrap with RAII
//  }
//
//  // output = input^T * input
//  void syrk(El::UpperOrLower uplo, const El::Matrix<double> &input,
//                               El::Matrix<double> &output)
//  {
//    // input: KxN matrix
//    // output = input^T * input: NxN matrix
//    assert(input.Width() == output.Width());
//    assert(output.Height() == output.Width());
//
//    CBLAS_LAYOUT layout = CblasColMajor;
//    CBLAS_UPLO Uplo = uplo == El::UpperOrLowerNS::UPPER ?  CblasUpper : CblasLower;
//    CBLAS_TRANSPOSE Trans = CblasTrans;
//    const CBLAS_INDEX N = input.Width();
//    const CBLAS_INDEX K = input.Height();
//    const double alpha = 1.0;
//    const double *A = input.LockedBuffer();
//    const CBLAS_INDEX lda = input.LDim();
//    const double beta = 0.0;
//    double *C = output.Buffer();
//    const CBLAS_INDEX ldc = output.LDim();
//    // C := alpha * A^T * A + beta * C = (in our case) A^T * A
//    cblas_dsyrk(layout, Uplo, Trans, N, K, alpha, A, lda, beta, C, ldc);
//  }
//
//  // Reconstruct matrix(i,j) element from its residues and save to fmpz_t &output.
//  void
//  restore_matrix_from_residues(const Residue_Matrices_Window<double> &window, size_t i,
//                     size_t j, Fmpz_Comb &comb, fmpz_t &output)
//  {
//    int sign = 1; // means that negative values are allowed
//    size_t num_primes = comb.num_primes;
//
//    std::vector<mp_limb_t> residues(num_primes);
//    for(size_t prime_index = 0; prime_index < num_primes; prime_index++)
//      {
//        double d = window.residues.at(prime_index)(i, j);
//        assert(abs(d) <= MAX_BLAS_DP_INT);
//        auto &mod = comb.mods.at(prime_index);
//        residues.at(prime_index) = double_to_uint32_t_residue(d, mod);
//      }
//
//    fmpz_multi_CRT_ui(output, residues.data(), comb.comb, comb.comb_temp,
//                      sign);
//  }
//}
//
//// Calculate Q := P^T P using Chinese Remainder Theorem and cblas_dsyrk()
//// P is normalized by columns and multiplied by 2^N (N=El::gmp::Precision()), see Matrix_Normalizer
//// Q is also normalized and can be restored
////
//// input_normalized_blocks are horizontal bands of P,
//// distributed among the ranks of shared_memory_comm (i.e. on a single cluster node)
////
//// NB: if you have several nodes, you have to reduce_scatter the output
//// to get a global Q matrix for COMM_WORLD
////
//// The function calculates contributions for all blocks from a single node
//// output is a DistMatrix
//// TODO: rearrange to   blas-like interface, e.g. bigint_syrk_crt_blas()
//// TODO combine residue windows and comb into Fmpz_CRT_Helper class
//void calculate_Block_Matrix_square(
//  El::mpi::Comm shared_memory_comm,
//  // Blocks stored for a given rank
//  const std::vector<El::DistMatrix<El::BigFloat>> &input_normalized_blocks,
//  // Indices of input_normalized_blocks in blocks_window
//  const std::vector<size_t> &block_indices_for_window,
//  Block_Residue_Matrices_Window<double> &blocks_window,
//  Residue_Matrices_Window<double> &output_residues_window, Fmpz_Comb &comb,
//  El::DistMatrix<El::BigFloat> &output)
//{
//  size_t width = output.Width();
//
//  assert(input_normalized_blocks.size() == block_indices_for_window.size());
//
//  assert(output.Height() == output.Width());
//  assert(output.Height() == width);
//  assert(blocks_window.width == width);
//
//  assert(El::mpi::Congruent(shared_memory_comm, blocks_window.Comm()));
//  assert(
//    El::mpi::Congruent(shared_memory_comm, output_residues_window.Comm()));
//  assert(El::mpi::Congruent(shared_memory_comm, output.DistComm()));
//
//  // Compute residues
//  for(size_t i = 0; i < input_normalized_blocks.size(); ++i)
//    {
//      // NB: block_indices should enumerate all blocks
//      // from all ranks in current node
//      size_t block_index = block_indices_for_window.at(i);
//      const auto &block = input_normalized_blocks.at(i);
//      assert(block.Width() == width);
//      compute_matrix_residues(block_index, block, comb, blocks_window);
//    }
//  // wait for all ranks to fill blocks_window
//  blocks_window.Fence();
//
//  // Square each residue matrix
//  auto comm = blocks_window.Comm();
//  for(size_t prime_index = 0; prime_index < comb.num_primes; ++prime_index)
//    {
//      if(prime_index % El::mpi::Size(comm) == El::mpi::Rank(comm))
//        {
//          const auto &input_matrix = blocks_window.residues.at(prime_index);
//          auto &output_matrix
//            = output_residues_window.residues.at(prime_index);
//          syrk(El::UpperOrLower::UPPER,input_matrix, output_matrix);
//        }
//    }
//  output_residues_window.Fence();
//
//  fmpz_t big_int_value;
//  fmpz_init(big_int_value);
//  El::BigFloat big_float_value;
//  for(int i = 0; i < output.LocalHeight(); ++i)
//    for(int j = 0; j < output.LocalWidth(); ++j)
//      {
//        int global_i = output.GlobalRow(i);
//        int global_j = output.GlobalCol(j);
//        restore_matrix_from_residues(output_residues_window, global_i,
//                                     global_j, comb, big_int_value);
//        fmpz_t_to_BigFloat(big_int_value, big_float_value);
//        output.SetLocal(i, j, big_float_value);
//      }
//
//  fmpz_clear(big_int_value); // TODO use RAII wrapper?
//
//  El::MakeSymmetric(El::UpperOrLowerNS::UPPER, output);
//
//  // TODO synchronize_Q() for all nodes
//  // TODO assert: after synchronize_Q Q(i,i)=2^2N
//}
//
//void bigint_syrk_blas(
//  BigInt_Shared_Memory_Syrk_Context &context, El::UpperOrLower uplo,
//  const std::vector<El::DistMatrix<El::BigFloat>> &bigint_input_matrix_blocks,
//  const std::vector<size_t> &block_indices_per_shared_memory_comm,
//  El::DistMatrix<El::BigFloat> &bigint_output)
//{
////  calculate_Block_Matrix_square(
////    context.shared_memory_comm, bigint_input_matrix_blocks,
////    block_indices_per_shared_memory_comm, context.input_block_residues_window,
////    context.output_residues_window, context.comb, bigint_output);
////  return;
//
//  size_t width = bigint_output.Width();
//
//  // TODO replace asserts (they are removed in release) with exception throws
//  assert(bigint_input_matrix_blocks.size() == block_indices_per_shared_memory_comm.size());
//
//  assert(bigint_output.Height() == bigint_output.Width());
//  assert(bigint_output.Height() == width);
//  assert(context.input_block_residues_window.width == width);
//
//  assert(El::mpi::Congruent(context.shared_memory_comm, context.input_block_residues_window.Comm()));
//  assert(
//    El::mpi::Congruent(context.shared_memory_comm, context.output_residues_window.Comm()));
//  assert(El::mpi::Congruent(context.shared_memory_comm, bigint_output.DistComm()));
//
//  // Compute residues
//  for(size_t i = 0; i < bigint_input_matrix_blocks.size(); ++i)
//    {
//      // NB: block_indices should enumerate all blocks
//      // from all ranks in current node
//      size_t block_index = block_indices_per_shared_memory_comm.at(i);
//      const auto &block = bigint_input_matrix_blocks.at(i);
//      assert(block.Width() == width);
//      compute_matrix_residues(block_index, block, context.comb, context.input_block_residues_window);
//    }
//  // wait for all ranks to fill blocks_window
//  context.input_block_residues_window.Fence();
//
//  // Square each residue matrix
//  auto comm = context.input_block_residues_window.Comm();
//  for(size_t prime_index = 0; prime_index < context.comb.num_primes; ++prime_index)
//    {
//      if(prime_index % El::mpi::Size(comm) == El::mpi::Rank(comm))
//        {
//          const auto &input_matrix = context.input_block_residues_window.residues.at(prime_index);
//          auto &output_matrix
//            = context.output_residues_window.residues.at(prime_index);
//          syrk(uplo,input_matrix, output_matrix);
//        }
//    }
//  context.output_residues_window.Fence();
//
//  fmpz_t big_int_value;
//  fmpz_init(big_int_value);
//  El::BigFloat big_float_value;
//  for(int i = 0; i < bigint_output.LocalHeight(); ++i)
//    for(int j = 0; j < bigint_output.LocalWidth(); ++j)
//      {
//        int global_i = bigint_output.GlobalRow(i);
//        int global_j = bigint_output.GlobalCol(j);
//
//        // Only half of output matrix is initialized, ignore the other one
//        if(uplo == El::UpperOrLowerNS::UPPER && global_i > global_j)
//          continue;
//        if(uplo == El::UpperOrLowerNS::LOWER && global_i < global_j)
//          continue;
//
//        restore_matrix_from_residues(context.output_residues_window, global_i,
//                                     global_j, context.comb, big_int_value);
//        fmpz_t_to_BigFloat(big_int_value, big_float_value);
//        bigint_output.SetLocal(i, j, big_float_value);
//      }
//
//  fmpz_clear(big_int_value); // TODO use RAII wrapper?
//
//  // TODO synchronize_Q() for all nodes
//  // TODO assert: after synchronize_Q Q(i,i)=2^2N
//}
