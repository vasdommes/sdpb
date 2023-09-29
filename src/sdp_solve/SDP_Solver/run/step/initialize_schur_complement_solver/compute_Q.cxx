#include "../../../../SDP.hxx"
#include "../../../../Block_Diagonal_Matrix.hxx"
#include "../../../../../Timers.hxx"

void initialize_Q_group(const SDP &sdp, const Block_Info &block_info,
                        const Block_Diagonal_Matrix &schur_complement,
                        Block_Matrix &schur_off_diagonal,
                        Block_Diagonal_Matrix &schur_complement_cholesky,
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

  schur_off_diagonal.blocks.clear();
  schur_off_diagonal.blocks.reserve(schur_complement_cholesky.blocks.size());
  
  for(size_t block = 0; block < schur_complement_cholesky.blocks.size();
      ++block)
    {
      auto &cholesky_timer(timers.add_and_start(
        "run.step.initializeSchurComplementSolver.Q.cholesky_"
        + std::to_string(block_info.block_indices[block])));
      schur_complement_cholesky.blocks[block] = schur_complement.blocks[block];

      Cholesky(El::UpperOrLowerNS::LOWER,
               schur_complement_cholesky.blocks[block]);
      cholesky_timer.stop();

      // schur_off_diagonal = L^{-1} B
      auto &solve_timer(timers.add_and_start(
        "run.step.initializeSchurComplementSolver.Q.solve_"
        + std::to_string(block_info.block_indices[block])));

      schur_off_diagonal.blocks.push_back(sdp.free_var_matrix.blocks[block]);
      El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
               El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
               El::BigFloat(1), schur_complement_cholesky.blocks[block],
               schur_off_diagonal.blocks[block]);

      solve_timer.stop();

      // Q = (L^{-1} B)^T (L^{-1} B) = schur_off_diagonal^T schur_off_diagonal
      auto &syrk_timer(timers.add_and_start(
        "run.step.initializeSchurComplementSolver.Q.syrk_"
        + std::to_string(block_info.block_indices[block])));
      El::DistMatrix<El::BigFloat> Q_group_view(
        El::View(Q_group, 0, 0, schur_off_diagonal.blocks[block].Width(),
                 schur_off_diagonal.blocks[block].Width()));
      El::Syrk(El::UpperOrLowerNS::UPPER, El::OrientationNS::TRANSPOSE,
               El::BigFloat(1), schur_off_diagonal.blocks[block],
               El::BigFloat(1), Q_group_view);
      syrk_timer.stop();
    }
}

void synchronize_Q(El::DistMatrix<El::BigFloat> &Q,
                   const El::DistMatrix<El::BigFloat> &Q_group,
                   Timers &timers);

void compute_Q(const SDP &sdp, const Block_Info &block_info,
               const Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               El::DistMatrix<El::BigFloat> &Q, const El::Grid &group_grid,
               Timers &timers)
{
  Scoped_Timer timer(timers, "run.step.initializeSchurComplementSolver.Q");
  El::DistMatrix<El::BigFloat> Q_group(Q.Height(), Q.Width(), group_grid);
  initialize_Q_group(sdp, block_info, schur_complement, schur_off_diagonal,
                     schur_complement_cholesky, Q_group, timers);
  synchronize_Q(Q, Q_group, timers);
}