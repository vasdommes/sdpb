#include "pmp_read/simpleboot/Mathematica_Simpleboot_Expression_Parser.hxx"
#include "pmp_read/simpleboot/data_provider/Abstract_Simpleboot_Data_Provider.hxx"
#include "unit_tests/util/util.hxx"

#include <catch2/catch_amalgamated.hpp>

using Test_Util::REQUIRE_Equal::diff;
namespace
{
  struct Simpleboot_Test_Data_Provider final
      : Abstract_Simpleboot_Data_Provider
  {
    explicit Simpleboot_Test_Data_Provider(const Simpleboot_Parameters &params)
        : Abstract_Simpleboot_Data_Provider(params)
    {}

    // Set default simpleboot parameters to arbitrary numbers
    // TODO: make them realistic.
    Simpleboot_Test_Data_Provider()
        : Simpleboot_Test_Data_Provider(
            Simpleboot_Parameters{1,
                                  2,
                                  3,
                                  10,
                                  {},
                                  (3 - 2 * sqrt(Boost_Float(2))) * 4,
                                  "",
                                  {}})
    {}

    [[nodiscard]] int get_index(const std::string &, const int L) const
    {
      return L;
    }

  protected:
    // Generate some arbitrary polynomial
    Polynomial blockF_lookup([[maybe_unused]] const std::string &stamp,
                             const int L, const int m, const int n) override
    {
      Polynomial poly(L + 5, 0);
      for(auto i = 0; i < poly.coefficients.size(); i++)
        {
          poly.coefficients.at(i) = i * n + m;
        }

      return poly;
    }
  };
}

TEST_CASE("simpleboot")
{
  SECTION("Mathematica_Simpleboot_Expression_Parser")
  {
    SECTION("no-crash-test")
    {
      INFO("Check that Mathematica_Simpleboot_Expression_Parser does not "
           "crash for the following inputs. "
           "NB: we do not check the parsing results are correct.");
      const auto provider = std::make_shared<Simpleboot_Test_Data_Provider>();
      Mathematica_Simpleboot_Expression_Parser parser(provider);

      const std::string input = GENERATE(
        "1*^2", "1", "1-2+3.3*4.5*^-2", "FS[\"stamp\",1,2,3,-4.5*^-7]",
        "PT[\"stamp\",1,2,3,0.4,-1.5*^-2]",
        R"(FS[\"stamp\",1,2,3,+1.5`200.23*^-7])",
        "F[\"Fespsigespsig\", 0, 1, 0, -0.4819999999999999840127884453977458178997\\\n\
03979492187499999999999999999999999999999999999999999999999999999999999999999\\\n\
99999999999999999999999999999999999999999999999999999999999999999999999999999\\\n\
999999999999999999`200.]");

      DYNAMIC_SECTION(input)
      {
        CAPTURE(input);
        const auto begin = input.c_str();
        const auto end = begin + input.size();
        MMA_ELEMENT result;
        parser.parse_element(begin, end, result);
        CAPTURE(result);
      }
    }
  }

  SECTION("Linear_Combination_Of_Mathematica_Functions")
  {
    INFO("Parse Linear_Combination_Of_Mathematica_Functions and evaluate it in parallel.");
    const auto provider = std::make_shared<Simpleboot_Test_Data_Provider>();
    Mathematica_Simpleboot_Expression_Parser<Simpleboot_Test_Data_Provider,
                                             /*evaluate_F0_F_FS=*/false>
      parser(provider);

    const std::string input = GENERATE( "FS[\"stamp\",1,2,3,-4.5*^-7]",
      "1.3 * FS[\"stamp\",1,2,3,-4.5*^-1] - F[\"stamp\",2,2,3,0.45] - 1.7*^-2 * F0[0.34,1,2]"
      );
    CAPTURE(input);
    const auto begin = input.c_str();
    const auto end = begin + input.size();

    MMA_ELEMENT result;
    parser.parse_element(begin, end, result);
    auto funcs = AS_MMA_ELEMENT_Linear_Combination_Of_Functions(result);

    INFO("NB: we only check that parallel and non-parallel computations give the same result.");
    INFO("TODO: check that this result is actually correct.");

    // compute everything on a single rank
    auto result_self
      = evaluate_parallel({funcs}, *provider, El::mpi::COMM_SELF);
    // compute different functions on different ranks
    auto result_world
      = evaluate_parallel({funcs}, *provider, El::mpi::COMM_WORLD);
    DIFF(result_self, result_world);
  }
}
