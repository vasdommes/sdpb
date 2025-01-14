#pragma once

#include "../Polynomial.hxx"

El::BigFloat
eval_summed(const std::vector<std::vector<Polynomial>> &summed_polynomials,
            const El::BigFloat &x);
