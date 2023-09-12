#pragma once

#include <flint/fmpz.h>
#include <vector>
#include <boost/noncopyable.hpp>

// adopted from FLINT, fmpz_mat/mul_blas.c
// TODO: rename arguments and add description to make the code readable
// TODO think which constructor we really want to use
struct Primes : boost::noncopyable
{
  fmpz_comb_t comb{};
  //  fmpz_comb_temp_struct *comb_temp{};
  fmpz_comb_temp_t comb_temp{};
  std::vector<mp_limb_t> primes;

  Primes() = delete;
  // TODO fix description, can be wrong
  // bits: Number of bits to store matrix multiplication result
  // k: number of additions.
  // If we multiply matrices A and B such that:
  //   max|A| < 2^Abits
  //   max|B| < 2^Bbits
  // then:
  //   sign = 0 if A and B are non-negative, 1 otherwise
  //   k = A.Width() = B.Height()
  //   bits = Abits + Bbits + bits(k) + sign
  Primes(flint_bitcnt_t bits, slong k);
  Primes(flint_bitcnt_t Abits, flint_bitcnt_t Bbits, int sign, slong k);
  ~Primes();

  size_t size() const;
};
