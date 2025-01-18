#pragma once

#include "Simpleboot_Parameters.hxx"
#include "pmp_read/simpleboot/mathematica_parse_util.hxx"

class Abstract_Simpleboot_Data_Provider
{
public:
  virtual ~Abstract_Simpleboot_Data_Provider() = default;

protected:
  explicit Abstract_Simpleboot_Data_Provider(
    const Simpleboot_Parameters &params);

public:
  // TODO store Simpleboot_Parameters field instead?
  const El::BigFloat dim;
  const El::BigFloat nu;
  const int kappa;
  const int maxderivs; // only used for interval positivity
  const std::map<std::string, El::BigFloat> var_map;

  // TODO this is constant, should we move it elsewhere?
  const Boost_Float r_crossing_4;

public:
  // Mathematica functions:
  // P[stamp, L, m, n, shift] : block derivative polynomial
  // F[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor
  // Fs[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor (shortened pole)

  // Fill Mathematica parser result

  void F0(const El::BigFloat &x, int m, int n, MMA_ELEMENT &result);

  void F(const std::string &stamp, int L, int m, int n, const El::BigFloat &x,
         MMA_ELEMENT &result);

  void FS(const std::string &stamp, int L, int m, int n, const El::BigFloat &x,
          MMA_ELEMENT &result);

  void PT(const std::string &stamp, int L, int m, int n, const El::BigFloat &a,
          const El::BigFloat &b, MMA_ELEMENT &result);

  void P(const std::string &stamp, int L, int m, int n,
         const El::BigFloat &shift, MMA_ELEMENT &result);

  // Directly evaluate functions

  El::BigFloat F0(const El::BigFloat &x, int m, int n);

  El::BigFloat
  F(const std::string &stamp, int L, int m, int n, const El::BigFloat &x);

  El::BigFloat
  FS(const std::string &stamp, int L, int m, int n, const El::BigFloat &x);

  Polynomial PT(const std::string &stamp, int L, int m, int n,
                const El::BigFloat &a, const El::BigFloat &b);

  Polynomial
  P(const std::string &stamp, int L, int m, int n, const El::BigFloat &shift);

  [[nodiscard]] virtual int get_index(const std::string &, const int L) const
    = 0;

private:
  // Prefactors

  [[nodiscard]] El::BigFloat Fprefactor(int L, const El::BigFloat &x) const;
  [[nodiscard]] El::BigFloat FSprefactor(int L, const El::BigFloat &x) const;

private:
  std::vector<std::vector<mpz_class>> binomial_cache;
  int binomial_cache_N = -1;

  // generate table of Binomial[m,n] with m,n<=N
  void init_binomial_coeff(int N);
  // return Binomial[m,n] , assuming m>=n
  mpz_class binomial_coeff_cached(int m, int n);
  void interval_transformation(std::vector<El::BigFloat> &coeff,
                               const El::BigFloat &a, const El::BigFloat &b,
                               int max_degree);

protected:
  virtual Polynomial
  blockF_lookup(const std::string &stamp, int L, int m, int n)
    = 0;
};
