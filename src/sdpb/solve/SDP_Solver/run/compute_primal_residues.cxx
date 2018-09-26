#include "constraint_matrix_weighted_sum.hxx"

// PrimalResidues = \sum_p A_p x[p] - X
//
// Inputs: sdp, x, X
// Output: PrimalResidues (overwritten)
//
void compute_primal_residues(const Block_Info &block_info, const SDP &sdp,
                             const Block_Vector &x,
                             const Block_Diagonal_Matrix &X,
                             Block_Diagonal_Matrix &primal_residues)
{
  constraint_matrix_weighted_sum(block_info, sdp, x, primal_residues);
  primal_residues -= X;
}