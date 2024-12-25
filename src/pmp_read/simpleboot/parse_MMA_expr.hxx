#pragma once

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/filesystem.hpp>

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>

#include <boost/serialization/vector.hpp>
#include <boost/algorithm/string.hpp>

#include <boost/math/tools/polynomial.hpp>

#include <set>

#include <vector>
#include <string>

#include <utility>

#include <variant>

#include <El.hpp>

#include <deque>
#include <list>
#include <map>

#include "pmp/Polynomial_Vector_Matrix.hxx"
#include "sdpb_util/Boost_Float.hxx"
#include "pmp/Polynomial.hxx"

////////////////////////////////////////   parse Mathematica expression //////////////////////////////////////

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

inline const char *ptr_MMA_begin;
inline const char *ptr_MMA_current;

#define MMA_PARSER_ERROR(...)                                                 \
  RUNTIME_ERROR("current ptr in MMA file : ",                                 \
                ptr_MMA_current - ptr_MMA_begin, "\n", __VA_ARGS__)


std::ostream &operator<<(std::ostream &os, const MMA_TOKEN &v);

using MMA_EXPR = std::variant<El::BigFloat, Polynomial>;
#define MMA_EXPR_Number 0
#define MMA_EXPR_Polynomial 1
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
#define SET_MMA_ELEMENT(var, value, T) var.emplace<MMA_ELEMENT_##T>(value)

std::ostream &operator<<(std::ostream &os, const MMA_ELEMENT &v);

const char *skip_space_from_left(const char *b, const char *e);

////////////////// built-in function and symbols ////////////////////////

namespace param
{
  inline El::BigFloat dim, nu;
  inline int kappa;
  inline int maxderivs; // only used for interval positivity
  inline std::string block_folder;
  inline std::vector<std::string> input_files;
  inline std::map<std::string, El::BigFloat> var_map;

  inline Boost_Float r_crossing_4;
  inline bool MPI_F_FS_parallelQ = false;
}

inline bool internal_print_Q = false;

inline std::map<std::pair<std::string, int>,
                std::vector<std::vector<std::vector<El::BigFloat>>>>
  blockF;
inline std::map<std::pair<std::string, int>, int> blockF_key2index;

void generate_blockF_key2index(
  const std::string &block_folder,
  std::map<std::pair<std::string, int>, int> &blockF_key2index);

// Parse Mathematica expressions

const char *
parse_MMA_expr(const char *begin, const char *end, MMA_ELEMENT &result);
void parse_MMA_symbol(const std::string &name, MMA_ELEMENT &result);
const char *parse_MMA_function(const std::string &name, const char *begin,
                               const char *end, MMA_ELEMENT &result);
const char *parse_MMA_token_as_int(const char *begin, const char *end,
                                   MMA_TOKEN &token, int &intnum);
const char *parse_MMA_token_as_float(const char *begin, const char *end,
                                     MMA_TOKEN &token, El::BigFloat &f);
const char *parse_MMA_token_as_string(const char *begin, const char *end,
                                      MMA_TOKEN &token, std::string &str);
const char *parse_MMA_check_op(const char *begin, const char *end,
                               MMA_TOKEN &token, char op);
const char *parse_MMA_get_op(const char *begin, const char *end,
                             MMA_TOKEN &token, char &op);

const char *
parse_MMA_element(const char *begin, const char *end, MMA_ELEMENT &result);

// for my purpose now, I only need
// P[stamp, L, m, n, shift] : block derivative polynomial
// F[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor
// Fs[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor (shortened pole)
const char *parse_MMA_function(const std::string &name, const char *begin,
                               const char *end, MMA_ELEMENT &result);

template <typename T>
const char *sb_parse_vector(const char *begin, const char *end,
                            std::vector<T> &result_vector);

// template const char *sb_parse_vector(const char *begin, const char *end,
//                                      std::vector<El::BigFloat> &result_vector);
// template const char *sb_parse_vector(const char *begin, const char *end,
//                                      std::vector<Boost_Float> &result_vector);

const char *sb_parse_polynomial(const char *begin, const char *end,
                                Polynomial &polynomial);

/////////////////////////////// parse damped rational constant //////////////////////////////////////////////////////////

const char *
sb_parse_damped_rational_constant(const char *constant_start, const char *end,
                                  Boost_Float &damped_rational_constant);

// Prefactors

El::BigFloat Fprefactor(int L, const El::BigFloat &x);
El::BigFloat FSprefactor(int L, const El::BigFloat &x);

// ----------- interval transformation  ---------------

void interval_transformation(std::vector<El::BigFloat> &coeff,
                             const El::BigFloat &a, const El::BigFloat &b,
                             int max_degree);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO
void mpi_counter_init(int init_value = 0);
int mpi_counter_get();
