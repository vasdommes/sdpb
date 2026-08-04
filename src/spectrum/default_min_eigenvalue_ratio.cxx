#include "pmp/PMP_Info.hxx"
#include "sdp_solve/read_text_block.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <boost/algorithm/string.hpp>

namespace fs = std::filesystem;

namespace
{
  El::BigFloat read_duality_gap_impl(const fs::path &solution_dir)
  {
    const auto out_txt_path = solution_dir / "out.txt";
    std::ifstream is(out_txt_path);
    ASSERT(is.good(), "Failed to open ", out_txt_path);
    std::string line;
    while(std::getline(is, line))
      {
        std::vector<std::string> tokens;
        boost::split(tokens, line, boost::is_any_of("=;"));
        for(auto &token : tokens)
          boost::trim(token);

        if(tokens.empty())
          break;
        if(tokens[0] != "dualityGap")
          continue;
        const auto duality_gap = El::BigFloat(tokens[1]);
        return duality_gap;
      }
    RUNTIME_ERROR("Cannot find dualityGap in ", out_txt_path);
  }

  El::BigFloat read_duality_gap(const fs::path &solution_dir, Timers &timers)
  {
    Scoped_Timer timer(timers, "read_duality_gap");
    El::BigFloat duality_gap;
    if(El::mpi::Rank() == 0)
      duality_gap = read_duality_gap_impl(solution_dir);
    El::mpi::Broadcast(duality_gap, 0, El::mpi::COMM_WORLD);
    return duality_gap;
  }
}

std::optional<El::BigFloat>
default_min_eigenvalue_ratio(const std::filesystem::path &solution_dir,
                             const PMP_Info &pmp_info,
                             const Verbosity &verbosity, Timers &timers)
{
  // --minEigenvalueRatio is not needed for 1x1 blocks.
  size_t max_block_dim = 0;
  for(const auto &block : pmp_info.blocks)
    max_block_dim = std::max(max_block_dim, block.dim);
  max_block_dim
    = El::mpi::AllReduce(max_block_dim, El::mpi::MAX, El::mpi::COMM_WORLD);
  if(max_block_dim <= 1)
    return std::nullopt;

  // Set --minEigenvalueRatio = sqrt(dualityGap)
  const auto duality_gap = read_duality_gap(solution_dir, timers);
  const auto min_eigenvalue_ratio = El::Sqrt(duality_gap);
  if(El::mpi::Rank() == 0)
    {
      if(verbosity >= Verbosity::regular)
        {
          El::Output("Setting --minEigenvalueRatio to sqrt(dualityGap)=",
                     min_eigenvalue_ratio);
        }
      ASSERT(min_eigenvalue_ratio >= 0 && min_eigenvalue_ratio <= 1,
             "--minEigenvalueRatio=", min_eigenvalue_ratio,
             " should be in range [0,1].");
    }
  // Wait for checks performed at rank=0
  El::mpi::Barrier();
  return min_eigenvalue_ratio;
}
