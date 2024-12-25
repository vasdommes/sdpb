#pragma once

#include "Simpleboot_Parameters.hxx"
#include "mathematica_parse_util.hxx"
#include "sdpb_util/Boost_Float.hxx"

#include <filesystem>

class Abstract_Simpleboot_Data_Provider
{
public:
  virtual ~Abstract_Simpleboot_Data_Provider() = default;

  // Mathematica functions:
  // P[stamp, L, m, n, shift] : block derivative polynomial
  // F[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor
  // Fs[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor (shortened pole)

  void
  F0(const El::BigFloat &x, const int m, const int n, MMA_ELEMENT &result);

  void F(const std::string &stamp, const int L, const int m, const int n,
         const El::BigFloat &x, MMA_ELEMENT &result);

  void FS(const std::string &stamp, const int L, const int m, const int n,
          const El::BigFloat &x, MMA_ELEMENT &result);

  void PT(const std::string &stamp, const int L, const int m, const int n,
          const El::BigFloat &a, const El::BigFloat &b, MMA_ELEMENT &result);

  void P(const std::string &stamp, int L, int m, int n,
         const El::BigFloat &shift, MMA_ELEMENT &result);

  // Prefactors

  El::BigFloat Fprefactor(int L, const El::BigFloat &x);
  El::BigFloat FSprefactor(int L, const El::BigFloat &x);

protected:
  virtual Polynomial
  blockF_lookup(const std::string &stamp, int L, int m, int n)
    = 0;

private:
  Boost_Float Pochhammer(const Boost_Float &alpha, const int64_t &n);

  // TODO move to matrix parser
  int current_matrix_max_polynomial_degree = -1;

public:
  // TODO store Simpleboot_Parameters field instead?
  const El::BigFloat dim;
  const El::BigFloat nu;
  const int kappa;
  const int maxderivs; // only used for interval positivity
  const std::map<std::string, El::BigFloat> var_map;

  // TODO this is constant, should we move it elsewhere?
  const Boost_Float r_crossing_4;
  // TODO remove:
  // bool MPI_F_FS_parallelQ = false;

private:
  std::vector<std::vector<mpz_class>> binomial_cache;
  int binomial_cache_N = -1;

  // generate table of Binomial[m,n] with m,n<=N
  int init_binomial_coeff(int N);
  // return Binomial[m,n] , assuming m>=n
  mpz_class binomial_coeff_cached(int m, int n);
  void interval_transformation(std::vector<El::BigFloat> &coeff,
                               const El::BigFloat &a, const El::BigFloat &b,
                               int max_degree);

protected:
  explicit Abstract_Simpleboot_Data_Provider(
    const Simpleboot_Parameters &params);
};
