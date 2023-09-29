#include "../../../../SDP.hxx"
#include "../../../../Block_Diagonal_Matrix.hxx"
#include "../../../../../Timers.hxx"
#include "bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "bigint_syrk/Matrix_Normalizer.hxx"

BigInt_Shared_Memory_Syrk_Context
initialize_syrk_context(const El::mpi::Comm &shared_memory_comm,
                        const Block_Info &block_info,
                        const Block_Matrix &schur_off_diagonal,
                        int block_width)
{
  auto num_ranks_per_node = El::mpi::Size(shared_memory_comm);

  // Collect block heights and block indices from all ranks in shared_memory_comm
  std::map<int, std::vector<size_t>> rank_to_global_block_indices;
  std::map<int, std::vector<int>> rank_to_block_heights;
  for(int rank = 0; rank < num_ranks_per_node; ++rank)
    {
      size_t num_blocks_in_rank = schur_off_diagonal.blocks.size();
      assert(num_blocks_in_rank == block_info.block_indices.size());
      El::mpi::Broadcast(num_blocks_in_rank, rank, shared_memory_comm);

      rank_to_global_block_indices.emplace(
        rank, std::vector<size_t>(num_blocks_in_rank));
      rank_to_block_heights.emplace(rank,
                                    std::vector<int>(num_blocks_in_rank));

      if(num_blocks_in_rank == 0)
        continue;
      if(El::mpi::Rank(shared_memory_comm) == rank)
        {
          rank_to_global_block_indices[rank] = block_info.block_indices;
          for(size_t block_index = 0; block_index < num_blocks_in_rank;
              ++block_index)
            {
              rank_to_block_heights[rank][block_index]
                = schur_off_diagonal.blocks[block_index].Height();
            }
        }
      assert(num_blocks_in_rank == rank_to_global_block_indices[rank].size());
      assert(num_blocks_in_rank == rank_to_block_heights[rank].size());

      El::mpi::Broadcast(rank_to_global_block_indices[rank].data(),
                         num_blocks_in_rank, rank, shared_memory_comm);
      El::mpi::Broadcast(rank_to_block_heights[rank].data(),
                         num_blocks_in_rank, rank, shared_memory_comm);
    }

  std::map<size_t, int> global_block_index_to_height;
  std::set<size_t> global_block_indices;
  for(int rank = 0; rank < num_ranks_per_node; ++rank)
    {
      for(size_t i = 0; i < rank_to_global_block_indices[rank].size(); ++i)
        {
          auto global_index = rank_to_global_block_indices[rank][i];
          global_block_indices.insert(global_index);

          auto height = rank_to_block_heights[rank][i];
          global_block_index_to_height.emplace(global_index, height);
        }
    }

  std::map<size_t, size_t> block_index_global_to_shmem;
  std::vector<int> shmem_block_index_to_height(global_block_indices.size());
  size_t curr_shmem_block_index = 0;
  for(size_t global_index : global_block_indices)
    {
      block_index_global_to_shmem.emplace(global_index,
                                          curr_shmem_block_index);
      shmem_block_index_to_height[curr_shmem_block_index]
        = global_block_index_to_height[global_index];
      curr_shmem_block_index++;
    }

  std::vector<size_t> block_index_local_to_shmem(
    block_info.block_indices.size());
  for(size_t local_index = 0; local_index < block_index_local_to_shmem.size();
      ++local_index)
    {
      size_t global_index = block_info.block_indices[local_index];
      size_t shmem_index = block_index_global_to_shmem[global_index];
      block_index_local_to_shmem[local_index] = shmem_index;
    }

  return BigInt_Shared_Memory_Syrk_Context(
    shared_memory_comm, El::gmp::Precision(), shmem_block_index_to_height,
    block_width, block_index_local_to_shmem);
}

// schur_off_diagonal = L^{-1} B
void initialize_schur_off_diagonal(
  const SDP &sdp, const Block_Info &block_info,
  const Block_Diagonal_Matrix &schur_complement,
  Block_Matrix &schur_off_diagonal,
  Block_Diagonal_Matrix &schur_complement_cholesky, Timers &timers)
{
  schur_off_diagonal.blocks.clear();
  schur_off_diagonal.blocks.reserve(schur_complement_cholesky.blocks.size());

  for(size_t block = 0; block < schur_complement_cholesky.blocks.size();
      ++block)
    {
      const auto &block_index_string
        = std::to_string(block_info.block_indices[block]);
      {
        Scoped_Timer cholesky_timer(
          timers, "run.step.initializeSchurComplementSolver.Q.cholesky_"
                    + block_index_string);
        schur_complement_cholesky.blocks[block]
          = schur_complement.blocks[block];

        Cholesky(El::UpperOrLowerNS::LOWER,
                 schur_complement_cholesky.blocks[block]);
      }

      // schur_off_diagonal = L^{-1} B
      Scoped_Timer solve_timer(
        timers, "run.step.initializeSchurComplementSolver.Q.solve_"
                  + block_index_string);

      schur_off_diagonal.blocks.push_back(sdp.free_var_matrix.blocks[block]);
      El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
               El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
               El::BigFloat(1), schur_complement_cholesky.blocks[block],
               schur_off_diagonal.blocks[block]);
    }
}

// Q = (L^{-1} B)^T (L^{-1} B) = schur_off_diagonal^T schur_off_diagonal
void initialize_Q_group(const El::mpi::Comm &shared_memory_comm,
                        const SDP &sdp, const Block_Info &block_info,
                        Block_Matrix &schur_off_diagonal,
                        El::DistMatrix<El::BigFloat> &Q_group, Timers &timers)
{
  // Explicitly deallocate the lower half of Q_group.  This
  // significantly reduces the total amount of memory required.
  El::Matrix<El::BigFloat> &local(Q_group.Matrix());
  for(int64_t row = 0; row < Q_group.Height(); ++row)
    for(int64_t column = 0; column < row; ++column)
      {
        if(Q_group.IsLocal(row, column))
          {
            mpf_clear(local(Q_group.LocalRow(row), Q_group.LocalCol(column))
                        .gmp_float.get_mpf_t());
            local(Q_group.LocalRow(row), Q_group.LocalCol(column))
              .gmp_float.get_mpf_t()[0]
              ._mp_d
              = nullptr;
          }
      }

  int block_width = Q_group.Width();
  BigInt_Shared_Memory_Syrk_Context context = initialize_syrk_context(
    shared_memory_comm, block_info, schur_off_diagonal, block_width);

  Matrix_Normalizer normalizer(schur_off_diagonal.blocks, block_width,
                               El::gmp::Precision(), shared_memory_comm);
  normalizer.normalize_and_shift_P_blocks(schur_off_diagonal.blocks);
  auto uplo = El::UPPER;
  context.bigint_syrk_blas(uplo, schur_off_diagonal.blocks, Q_group, timers);
  normalizer.restore_P_blocks(schur_off_diagonal.blocks);

  // TODO: synchronize_Q, check that Q(i,i)==2^2N, then restore
  normalizer.restore_Q(uplo, Q_group);
}

void synchronize_Q(El::DistMatrix<El::BigFloat> &Q,
                   const El::DistMatrix<El::BigFloat> &Q_group,
                   Timers &timers);

void compute_Q(const SDP &sdp, const Block_Info &block_info,
               const Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers)
{
  Scoped_Timer timer(timers,
                     "run.step.initializeSchurComplementSolver.compute_Q");

  initialize_schur_off_diagonal(sdp, block_info, schur_complement,
                                schur_off_diagonal, schur_complement_cholesky,
                                timers);

  MPI_Comm shared_memory_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shared_memory_comm);
  const El::Grid grid(shared_memory_comm);
  El::DistMatrix<El::BigFloat> Q_group(Q.Height(), Q.Width(), grid);
  initialize_Q_group(shared_memory_comm, sdp, block_info, schur_off_diagonal,
                     Q_group, timers);
  synchronize_Q(Q, Q_group, timers);
}