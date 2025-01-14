#include "integration_tests/common.hxx"

// Realistic end-to-end test for sdp2input + sdpb
// JSON input taken from  "SingletScalar_cT_test_nmax6" and
// "SingletScalarAllowed_test_nmax6"
// https://gitlab.com/davidsd/scalars-3d/-/blob/master/src/Projects/Scalars3d/SingletScalar2020.hs
// Test data is generated with SDPB 2.5.1 on Caltech cluster.
// Note that on different machines results can vary due to rounding errors,
// depending on GMP/MPFR version etc.

namespace
{
  void
  end_to_end_test(const std::string &name,
                  const std::string &default_sdpb_args,
                  const std::string &sdp_format, int num_procs, int precision,
                  int sdp_zip_diff_precision, int sdpb_output_diff_precision,
                  const std::vector<std::string> &out_txt_keys = {})
  {
    auto format_description = sdp_format.empty() ? "default(bin)" : sdp_format;
    CAPTURE(sdp_format);
    CAPTURE(format_description);

    Test_Util::Test_Case_Runner runner(name + "/" + sdp_format);
    const auto &data_dir = runner.data_dir.parent_path();
    const auto &output_dir = runner.output_dir;

    auto sdp_orig_zip = (data_dir / ("sdp.orig.zip")).string();
    auto sdp_zip = (output_dir / "sdp.zip").string();

    // pvm2sdp or sdp2input
    {
      if(exists(data_dir / "pvm.xml"))
        {
          INFO("run pvm2sdp");
          runner.create_nested("pvm2sdp").mpi_run(
            {"build/pvm2sdp", sdp_format, std::to_string(precision),
             (data_dir / "pvm.xml").string(), sdp_zip},
            {}, num_procs);
        }
      else
        {
          INFO("run sdp2input");
          Test_Util::Test_Case_Runner::Named_Args_Map args{
            {"--input", (data_dir / "json" / "file_list.nsv").string()},
            {"--output", sdp_zip},
            {"--outputFormat", sdp_format},
            {"--precision", std::to_string(precision)}};

          if(!sdp_format.empty())
            args["--outputFormat"] = sdp_format;

          runner.create_nested("sdp2input")
            .mpi_run({"build/sdp2input"}, args, num_procs);
        }

      // sdp2input runs with --precision=<precision>
      // We check test output up to lower precision=<sdp_zip_diff_precision>
      // in order to neglect unimportant rounding errors
      Test_Util::REQUIRE_Equal::diff_sdp_zip(
        sdp_zip, sdp_orig_zip, precision, sdp_zip_diff_precision,
        runner.create_nested("sdp.zip.diff"));
    }

    // sdpb
    {
      Test_Util::Test_Case_Runner::Named_Args_Map args{
        {"--procsPerNode", std::to_string(num_procs)},
        {"--precision", std::to_string(precision)},
        {"--sdpDir", sdp_zip},
        {"--outDir", (output_dir / "out").string()},
        {"--checkpointDir", (output_dir / "ck").string()}};
      runner.create_nested("sdpb/run")
        .mpi_run({"build/sdpb", default_sdpb_args}, args, num_procs);

      // SDPB runs with --precision=<precision>
      // We check test output up to lower precision=<sdpb_output_diff_precision>
      // in order to neglect unimportant rounding errors
      Test_Util::REQUIRE_Equal::diff_sdpb_output_dir(
        output_dir / "out", data_dir / "out", precision,
        sdpb_output_diff_precision, {}, out_txt_keys);
    }
  }
}

TEST_CASE("end-to-end_tests")
{
  INFO("End-to-end tests for pvm2sdp/sdp2input + sdpb");
  INFO("On different machines results can vary due to rounding errors, "
       "depending on GMP/MPFR version etc");
  int num_procs = 6;
  int precision = 768;
  int sdp_zip_diff_precision = 608;
  std::string name = "end-to-end_tests";

  SECTION("dfibo-0-0-j=3-c=3.0000-d=3-s=6")
  {
    INFO("pvm2sdp+sdpb test for https://github.com/davidsd/sdpb/issues/124");
    INFO("sdp.zip contains block with empty bilinear_bases_odd, "
         "which caused a bug.");
    INFO("Test data from Harvard cluster, gmp/6.2.1 mpfr/4.2.0");
    name += "/dfibo-0-0-j=3-c=3.0000-d=3-s=6";
    std::string default_sdpb_args
      = "--findDualFeasible --findPrimalFeasible "
        "--initialMatrixScalePrimal 1e10 --initialMatrixScaleDual 1e10 "
        "--maxComplementarity 1e30 --dualErrorThreshold 1e-10 "
        "--primalErrorThreshold 1e-153 --maxRuntime 259200 "
        "--checkpointInterval 3600 --maxIterations 1000 "
        "--feasibleCenteringParameter=0.1 --infeasibleCenteringParameter=0.3 "
        "--stepLengthReduction=0.7";
    int sdpb_output_diff_precision = 600;
    for(std::string sdp_format : {"", "bin", "json"})
      {
        // for sdp_format="" pvm2sdp will use bin format (by default)
        DYNAMIC_SECTION((sdp_format.empty() ? "default(bin)" : sdp_format))
        {
          end_to_end_test(name, default_sdpb_args, sdp_format, num_procs,
                          precision, sdp_zip_diff_precision,
                          sdpb_output_diff_precision);
        }
      }
  }

  SECTION("SingletScalar_nmax6")
  {
    INFO("sdp2input+sdpb tests based on "
         "https://gitlab.com/davidsd/scalars-3d/-/blob/master/src/Projects/"
         "Scalars3d/SingletScalar2020.hs");
    INFO("Test data is generated with SDPB 2.5.1 on Caltech cluster.");
    SECTION("SingletScalar_cT_test_nmax6/primal_dual_optimal")
    {
      INFO("SingletScalar_cT_test_nmax6 from "
           "https://gitlab.com/davidsd/scalars-3d/-/blob/master/src/Projects/"
           "Scalars3d/SingletScalar2020.hs");
      INFO("SDPB should find primal-dual optimal solution.");
      name += "/SingletScalar_cT_test_nmax6/primal_dual_optimal";
      std::string default_sdpb_args
        = "--checkpointInterval 3600 --maxRuntime 1340 "
          "--dualityGapThreshold 1.0e-30 --primalErrorThreshold 1.0e-30 "
          "--dualErrorThreshold 1.0e-30 --initialMatrixScalePrimal 1.0e20 "
          "--initialMatrixScaleDual 1.0e20 --feasibleCenteringParameter 0.1 "
          "--infeasibleCenteringParameter 0.3 --stepLengthReduction 0.7 "
          "--maxComplementarity 1.0e100 --maxIterations 1000 --verbosity 1 "
          "--procGranularity 1 --writeSolution y";
      int sdpb_output_diff_precision = 600;
      // This test is slow, we don't want to run it twice
      // json/bin correctness is checked by other tests below,
      // so we use only binary SDP here
      end_to_end_test(name, default_sdpb_args, "bin", num_procs, precision,
                      sdp_zip_diff_precision, sdpb_output_diff_precision);
    }

    SECTION("SingletScalarAllowed_test_nmax6")
    {
      INFO("SingletScalarAllowed_test_nmax6 from "
           "https://gitlab.com/davidsd/scalars-3d/-/blob/master/src/Projects/"
           "Scalars3d/SingletScalar2020.hs");
      name += "/SingletScalarAllowed_test_nmax6";
      std::string default_sdpb_args
        = "--checkpointInterval 3600 --maxRuntime 1341 "
          "--dualityGapThreshold 1.0e-30 --primalErrorThreshold 1.0e-200 "
          "--dualErrorThreshold 1.0e-200 --initialMatrixScalePrimal 1.0e20 "
          "--initialMatrixScaleDual 1.0e20 --feasibleCenteringParameter 0.1 "
          "--infeasibleCenteringParameter 0.3 --stepLengthReduction 0.7 "
          "--maxComplementarity 1.0e100 --maxIterations 1000 --verbosity 1 "
          "--procGranularity 1 --writeSolution y "
          "--detectPrimalFeasibleJump --detectDualFeasibleJump";
      int sdpb_output_diff_precision = 610;

      SECTION("primal_feasible_jump")
      {
        INFO("SDPB should detect primal feasible jump.");
        name += "/primal_feasible_jump";
        std::vector<std::string> out_txt_keys
          = {"terminateReason", "primalObjective", "dualObjective",
             "dualityGap", "dualError"};
        for(auto &sdp_format : {"bin", "json"})
          {
            DYNAMIC_SECTION(sdp_format)
            {
              end_to_end_test(name, default_sdpb_args, sdp_format, num_procs,
                              precision, sdp_zip_diff_precision,
                              sdpb_output_diff_precision, out_txt_keys);
            }
          }
      }
      SECTION("dual_feasible_jump")
      {
        INFO("SDPB should detect dual feasible jump.");
        name += "/dual_feasible_jump";
        std::vector<std::string> out_txt_keys
          = {"terminateReason", "primalObjective", "dualObjective",
             "dualityGap", "primalError"};
        for(auto &sdp_format : {"bin", "json"})
          {
            DYNAMIC_SECTION(sdp_format)
            {
              end_to_end_test(name, default_sdpb_args, sdp_format, num_procs,
                              precision, sdp_zip_diff_precision,
                              sdpb_output_diff_precision, out_txt_keys);
            }
          }
      }
    }
  }
}
