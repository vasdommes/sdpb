#include "../../../../SDP_Solver.hxx"

// min(gamma \alpha(M, dM), 1), where \alpha(M, dM) denotes the
// largest positive real number such that M + \alpha dM is positive
// semidefinite.
//
// \alpha(M, dM) is computed with a Cholesky decomposition M = L L^T.
// The eigenvalues of M + \alpha dM are equal to the eigenvalues of 1
// + \alpha L^{-1} dM L^{-T}.  The correct \alpha is then -1/lambda,
// where lambda is the smallest eigenvalue of L^{-1} dM L^{-T}.
//
// Inputs:
// - MCholesky = L, the Cholesky decomposition of M (M itself is not needed)
// - dM, a Block_Diagonal_Matrix with the same structure as M
// Workspace:
// - MInvDM (NB: overwritten when computing minEigenvalue)
// - eigenvalues, a Vector of eigenvalues for each block of M
// Output:
// - min(\gamma \alpha(M, dM), 1) (returned)

// A := L^{-1} A L^{-T}
void lower_triangular_inverse_congruence(const Block_Diagonal_Matrix &L,
                                         Block_Diagonal_Matrix &A);

El::BigFloat min_eigenvalue(Block_Diagonal_Matrix &A);

El::BigFloat step_length(const Block_Diagonal_Matrix &MCholesky,
                         const Block_Diagonal_Matrix &dM,
                         const El::BigFloat &gamma,
                         const std::string &timer_name,
                         Timers &timers)
{
  auto &step_length_timer(
        timers.add_and_start(timer_name));
  // MInvDM = L^{-1} dM L^{-T}, where M = L L^T
  Block_Diagonal_Matrix MInvDM(dM);
  //  El::Output(El::mpi::Rank()," MInvDM(dM) ", timer_name);
  lower_triangular_inverse_congruence(MCholesky, MInvDM);
  El::Output(El::mpi::Rank(), " lower_triangular_inverse_congruence ",
             timer_name);
  const El::BigFloat lambda(min_eigenvalue(MInvDM));
  El::Output(El::mpi::Rank(), " min_eigenvalue ", timer_name);
  step_length_timer.stop();
  El::Output(El::mpi::Rank(), " finished ", timer_name);
  if(lambda > -gamma)
    {
      return 1;
    }
  else
    {
      return -gamma / lambda;
    }
}
