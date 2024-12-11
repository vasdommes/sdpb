// TODO move to unit_tests

#include "parse_MMA_expr.hxx"
#include "pmp/Polynomial.hxx"
#include "sdpb_util/Boost_Float.hxx"

#include <El.hpp>
#include <string>

void test_mpi()
{
  mpi_counter_init();

  std::vector<int> matrix_indices;

  int counter;
  do
    {
      counter = mpi_counter_get();
      matrix_indices.push_back(counter);
      std::cout << "current rank : " << El::mpi::Rank()
                << " get jobid=" << counter << "\n";
  } while(counter < 200);

  std::cout << "[Rank=" << El::mpi::Rank() << "] "
            << "matrices_valid_indices=";
  for(auto i : matrix_indices)
    std::cout << i << " ";
  std::cout << "\n";

  return;
}

void test_PT()
{
  Polynomial poly;
  poly.coefficients = {2, 3.1, 3.2, -10.32};

  std::cout << "test interval transformation : p(x) = " << poly << "\n";

  interval_transformation(poly.coefficients, 2.3, 5.3, 20);

  std::cout << "transformed p = " << poly << "\n";
}

void test_gmp_mpfr()
{
  using namespace param;

  std::cout << "------- test gmp begin ----------------\n";

  // set constant
  //r_crossing_4 = (3 - 2 * sqrt(Boost_Float(2))) * 4;

  Boost_Float rstar4 = (3 - 2 * sqrt(Boost_Float(2))) * 4;

  std::cout << "mpfr_get_default_prec() =" << mpfr_get_default_prec() << "\n";
  std::cout << "Boost_Float default precision="
            << Boost_Float::default_precision() << "\n";

  std::cout << std::setprecision(200) << std::fixed;
  std::cout << "rstar4=" << rstar4
            << ", prec=" << mpfr_get_prec(rstar4.backend().data()) << "\n";

  std::cout << "r_crossing_4=" << r_crossing_4
            << ", prec=" << mpfr_get_prec(r_crossing_4.backend().data())
            << "\n";

  r_crossing_4 = rstar4;

  std::cout << "r_crossing_4=" << r_crossing_4
            << ", prec=" << mpfr_get_prec(r_crossing_4.backend().data())
            << "\n";

  std::cout << "dim=" << dim << ", prec=" << dim.Precision() << "\n";
  std::cout << "nu=" << nu << ", prec=" << nu.Precision() << "\n";

  FSprefactor(2, "3.14");
  FSprefactor(1, "3.145");

  Fprefactor(2, "3.14");
  Fprefactor(1, "3.145");

  std::cout << "FSprefactor(0, 0.412)=" << FSprefactor(0, "0.412") << "\n";
  std::cout << "FSprefactor(2, 0)=" << FSprefactor(2, "0") << "\n";

  std::cout << "--------- test gmp end ----------------\n";

  exit(0);
  return;
}

//std::string test_str = "2.3 - 0.3 P[stamp, 1, 1, 2, -0.134] + P[stamp, 1, 1, 2, 3.1*sym+13.2`200]/(1+3) - P[stamp, 0, 1, 2, 0.134] +1.3 , 2+3  ";
//std::string test_str = " 1.2+3 (2-7) + 2 3 - (2.4 * 7)/(-3+2/3*4) + Fs[stamp, 1, 1, 2, 0.134, 3, 17] , 2+3  ";
//std::string test_str = " -1.2+3/2*3-1.1*1.2 3.4 + 4 , 1+3";
//std::string test_str = "-2.3 + 2.3 Fs[stamp, 1, 1, 2, 0.134, 3, 17] , 2+3  ";
//std::string test_str = "-2.3 + 2.3";

std::string test_str
  = "F[\"Fespsigespsig\", 0, 1, 0, -0.4819999999999999840127884453977458178997\\\n\
03979492187499999999999999999999999999999999999999999999999999999999999999999\\\n\
99999999999999999999999999999999999999999999999999999999999999999999999999999\\\n\
999999999999999999`200.]";

void test_parse_MMA_expr()
{
  std::cout << "test=" << test_str << "\n";

  const char *begin = test_str.c_str();
  const char *end = test_str.c_str() + test_str.size();

  MMA_ELEMENT result;
  parse_MMA_expr(begin, end, result);

  std::cout << "result=" << result << "\n";

  return;
}

void test_MMA_element()
{
  const char *begin = test_str.c_str();
  const char *end = test_str.c_str() + test_str.size();

  std::cout.precision(15);

  const char *pstr = begin;
  int i = 1;
  while(pstr != end && i <= 5000)
    {
      MMA_ELEMENT element;

      std::cout << "current str = " << pstr << "\n";

      pstr = parse_MMA_element(pstr, end, element);
      std::cout << "Find element #" << i << " : " << element << "\n";

      if(element.index() == 0)
        break;

      i++;
    }
  exit(0);
  return;
}
