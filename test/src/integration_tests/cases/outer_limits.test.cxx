#include "integration_tests/common.hxx"

#include <filesystem>

namespace fs = std::filesystem;

TEST_CASE("outer_limits")
{
  INFO("Simple outer_limits test");
  Test_Util::Test_Case_Runner runner("outer_limits");

  fs::path data_dir = runner.data_dir;
  fs::path output_dir = runner.output_dir;

  Test_Util::Test_Case_Runner::Named_Args_Map args{
    {"--functions", (data_dir / "toy_functions.json").string()},
    {"--out", (output_dir / "toy_functions_out.json").string()},
    {"--checkpointDir", (output_dir / "ck").string()},
    {"--points", (data_dir / "toy_functions_points.json").string()},
    {"--precision", "128"},
    {"--dualityGapThreshold", "1e-10"},
    {"--primalErrorThreshold", "1e-10"},
    {"--dualErrorThreshold", "1e-10"},
    {"--initialMatrixScalePrimal", "1e1"},
    {"--initialMatrixScaleDual", "1e1"},
    {"--maxIterations", "1000"},
    {"--verbosity", "1"},
  };

  runner.create_nested("run").mpi_run({"build/outer_limits"}, args);

  auto out = output_dir / "toy_functions_out.json";
  auto out_orig = data_dir / "toy_functions_out_orig.json";
  Test_Util::REQUIRE_Equal::diff_outer_limits(out, out_orig, 128, 128);
}
