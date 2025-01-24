#pragma once

// Utility functions for Mathematica_Simpleboot_Expression_Parser

#include "Linear_Combination_Of_Mathematica_Functions.hxx"
#include "pmp/Polynomial.hxx"
#include "sdpb_util/Boost_Float.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>
#include <utility>
#include <variant>

using MMA_TOKEN = std::variant<std::monostate, int, El::BigFloat, char,
                               std::string, std::string>;
#define MMA_TOKEN_Invalid 0
#define MMA_TOKEN_Integer 1
#define MMA_TOKEN_Real 2
#define MMA_TOKEN_Operator 3
#define MMA_TOKEN_Symbol 4
#define MMA_TOKEN_String 5

#define AS_MMA_TOKEN(var, T) std::get<MMA_TOKEN_##T>(var)
#define SET_MMA_TOKEN(var, value, T) var.emplace<MMA_TOKEN_##T>(value)

using MMA_EXPR = std::variant<El::BigFloat, Polynomial,
                              Linear_Combination_Of_Mathematica_Functions>;
#define MMA_EXPR_Number 0
#define MMA_EXPR_Polynomial 1
#define MMA_EXPR_Linear_Combination_Of_Functions 2
#define AS_MMA_EXPR(var, T) std::get<MMA_EXPR_##T>(var)
#define SET_MMA_EXPR(var, value, T) var.emplace<MMA_EXPR_##T>(value)

using MMA_ELEMENT = std::variant<std::monostate, MMA_EXPR, char>;
#define MMA_ELEMENT_Invalid 0
#define MMA_ELEMENT_Expression 1
#define MMA_ELEMENT_Operator 2
#define AS_MMA_ELEMENT(var, T) std::get<MMA_ELEMENT_##T>(var)
#define AS_MMA_ELEMENT_Number(var)                                            \
  std::get<MMA_EXPR_Number>(std::get<MMA_ELEMENT_Expression>(var))
#define AS_MMA_ELEMENT_Polynomial(var)                                        \
  std::get<MMA_EXPR_Polynomial>(std::get<MMA_ELEMENT_Expression>(var))
#define AS_MMA_ELEMENT_Linear_Combination_Of_Functions(var)                   \
  std::get<MMA_EXPR_Linear_Combination_Of_Functions>(                         \
    std::get<MMA_ELEMENT_Expression>(var))
#define SET_MMA_ELEMENT(var, value, T) var.emplace<MMA_ELEMENT_##T>(value)

std::ostream &operator<<(std::ostream &os, const MMA_ELEMENT &v);
std::ostream &operator<<(std::ostream &os, const MMA_TOKEN &v);

std::string to_string(const MMA_ELEMENT &v);
std::string to_string(const MMA_TOKEN &v);

// Helper function used by Json_Simpleboot_Float_Parser and Json_Simpleboot_Polynomial_Parser.
// We had to move from_MMA_element() outside of class
// because C++ does not allow partial specialization for member functions
template <class TResult>
TResult from_MMA_element(const MMA_ELEMENT &element) = delete;

template <> inline El::BigFloat from_MMA_element(const MMA_ELEMENT &element)
{
  return AS_MMA_ELEMENT_Number(element);
}
template <> inline Boost_Float from_MMA_element(const MMA_ELEMENT &element)
{
  return to_Boost_Float(AS_MMA_ELEMENT_Number(element));
}
template <> inline Polynomial from_MMA_element(const MMA_ELEMENT &element)
{
  return AS_MMA_ELEMENT_Polynomial(element);
}
