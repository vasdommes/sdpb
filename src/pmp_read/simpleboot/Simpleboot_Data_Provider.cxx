#include "Simpleboot_Data_Provider.hxx"

#include "mathematica_parse_util.hxx"

// (-1)^(m + n)*2^(1 + m + n - 2*x)*Poch[1 - m + x, m]*Poch[1 - n + x, n]
void Abstract_Simpleboot_Data_Provider::F0(const El::BigFloat &x, const int m,
                                           const int n, MMA_ELEMENT &result)
{
  // TODO remove
  // if(MPI_F_FS_parallelQ == true && El::mpi::Rank() != 0)
  //   {
  //     result = El::BigFloat(0);
  //     return;
  //   }

  Boost_Float BF_x = to_Boost_Float(x);
  Boost_Float BF_result = pow(2, 1 + m + n - 2 * BF_x)
                          * Pochhammer(1 - m + BF_x, m)
                          * Pochhammer(1 - n + BF_x, n);
  if((m + n) % 2 != 0)
    BF_result *= -1;

  result = to_BigFloat(BF_result);
}

void Abstract_Simpleboot_Data_Provider::F(const std::string &stamp,
                                          const int L, const int m,
                                          const int n, const El::BigFloat &x,
                                          MMA_ELEMENT &result)
{
  // TODO remove
  // if(MPI_F_FS_parallelQ == true
  //    && MPI_stamp_spin_to_rank(stamp, L) != El::mpi::Rank())
  //   {
  //     result = El::BigFloat(0);
  //     return;
  //   }

  Polynomial poly = blockF_lookup(stamp, L, m, n);

  result = poly(x) * Fprefactor(L, x);
}

void Abstract_Simpleboot_Data_Provider::FS(const std::string &stamp,
                                           const int L, const int m,
                                           const int n, const El::BigFloat &x,
                                           MMA_ELEMENT &result)
{
  // TODO remove
  // if(MPI_F_FS_parallelQ == true
  //    && MPI_stamp_spin_to_rank(stamp, L) != El::mpi::Rank())
  //   {
  //     result = El::BigFloat(0);
  //     return;
  //   }

  Polynomial poly = blockF_lookup(stamp, L, m, n);

  result = poly(x) * FSprefactor(L, x);
}

void Abstract_Simpleboot_Data_Provider::PT(const std::string &stamp,
                                           const int L, const int m,
                                           const int n, const El::BigFloat &a,
                                           const El::BigFloat &b,
                                           MMA_ELEMENT &result)
{
  Polynomial poly = blockF_lookup(stamp, L, m, n);

  // TODO checking and updating current_matrix_max_polynomial_degree
  // makes the object non-reusable for different matrices!
  // We should move this logic elsewhere
  if(current_matrix_max_polynomial_degree == -1)
    current_matrix_max_polynomial_degree = poly.degree() + (maxderivs - m - n);

  if(current_matrix_max_polynomial_degree
     != poly.degree() + (maxderivs - m - n))
    RUNTIME_ERROR("inconsistent polynomial degree : prediction from stamp=",
                  stamp, ", L=", L, ", m=", m, ", n=", n, " is ",
                  poly.degree() + (maxderivs - m - n),
                  ", while previous prediction is ", maxderivs);

  if(maxderivs < m + n)
    RUNTIME_ERROR("incorrect maxderivs in the param file: maxderivs=",
                  maxderivs);

  interval_transformation(poly.coefficients, a, b,
                          current_matrix_max_polynomial_degree);

  result = std::move(poly);
}

void Abstract_Simpleboot_Data_Provider::P(const std::string &stamp, int L,
                                          int m, int n,
                                          const El::BigFloat &shift,
                                          MMA_ELEMENT &result)
{
  Polynomial poly = blockF_lookup(stamp, L, m, n);

  if(shift != El::BigFloat(0))
    {
      poly.shift(shift);
    }
  result = std::move(poly);
}

El::BigFloat
Abstract_Simpleboot_Data_Provider::Fprefactor(int L, const El::BigFloat &x)
{
  const auto &order = kappa;
  El::BigFloat denominator = 1;
  El::BigFloat Delta = x + (L + dim - 2);

  for(int64_t k = 1; k <= order; ++k)
    denominator *= Delta - (-k - L + 1);
  for(int64_t k = 1; 2 * k <= order; ++k)
    denominator *= Delta - (nu + 1 - k);
  for(int64_t k = 1; k <= std::min(order, L); ++k)
    denominator *= Delta - (1 + 2 * nu + L - k);

  El::BigFloat numerator_BigFloat(
    to_BigFloat(pow(r_crossing_4, to_Boost_Float(Delta))));

  return numerator_BigFloat / denominator;
}

El::BigFloat
Abstract_Simpleboot_Data_Provider::FSprefactor(int L, const El::BigFloat &x)
{
  const auto &order = kappa;
  El::BigFloat denominator = 1;
  El::BigFloat Delta = x + (L + dim - 2);

  for(int64_t k = 2; k <= order; k = k + 2)
    denominator *= Delta - (-k - L + 1);
  for(int64_t k = 1; 2 * k <= order; ++k)
    denominator *= Delta - (nu + 1 - k);
  for(int64_t k = 2; k <= std::min(order, L); k = k + 2)
    denominator *= Delta - (1 + 2 * nu + L - k);

  El::BigFloat numerator_BigFloat(
    to_BigFloat(pow(r_crossing_4, to_Boost_Float(Delta))));

  return numerator_BigFloat / denominator;
}

Boost_Float
Abstract_Simpleboot_Data_Provider::Pochhammer(const Boost_Float &alpha,
                                              const int64_t &n)
{
  Boost_Float result(1);
  for(int64_t kk = 0; kk < n; ++kk)
    {
      result *= alpha + kk;
    }
  return result;
}

int Abstract_Simpleboot_Data_Provider::init_binomial_coeff(int N)
{
  if(N < binomial_cache_N)
    return 0;
  binomial_cache_N = N;
  binomial_cache.resize(N + 1);
  for(int m = 0; m <= N; m++)
    {
      binomial_cache[m].resize(m + 1);

      mpz_class binomial_coeff = 1;
      for(int n = 0; n <= m; n++)
        {
          // store Binomial[m,n]
          binomial_cache[m][n] = binomial_coeff;

          // Binomial[m,n+1]=Binomial[m,n] * (m-n)/(1+n)
          binomial_coeff = (binomial_coeff * (m - n)) / (n + 1);
        }
    }
  return 1;
}

mpz_class
Abstract_Simpleboot_Data_Provider::binomial_coeff_cached(int m, int n)
{
  return binomial_cache[m][n];
}

void Abstract_Simpleboot_Data_Provider::interval_transformation(
  std::vector<El::BigFloat> &coeff, const El::BigFloat &a,
  const El::BigFloat &b, int max_degree)
{
  int N = coeff.size() - 1; // degree of the polynomial
  int M = max_degree; // maximum degree of the polynomials in current matrix

  std::vector<El::BigFloat> new_coefficients(M + 1, 0);

  init_binomial_coeff(M);

  for(int n = 0; n <= N; n++)
    {
      El::BigFloat a_pow = to_BigFloat(pow(to_Boost_Float(a), n));
      El::BigFloat b_pow = 1;
      for(int m = 0; m <= n; m++)
        {
          for(int k = 0; k <= M - n; k++)
            {
              new_coefficients[k + m].gmp_float
                += coeff[n].gmp_float * a_pow.gmp_float * b_pow.gmp_float
                   * binomial_coeff_cached(n, m)
                   * binomial_coeff_cached(M - n, k);
            }
          a_pow = a_pow / a;
          b_pow = b_pow * b;
        }
    }

  coeff.assign(new_coefficients.begin(), new_coefficients.end());
}
