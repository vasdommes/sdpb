#pragma once

#include "sdpb_util/assert.hxx"
#include "sdpb_util/boost_serialization.hxx"

#include <El.hpp>
#include <type_traits>
#include <utility>
#include <variant>

// Mathematica functions

struct Mathematica_Function_F0
{
  Mathematica_Function_F0(El::BigFloat x, const int m, const int n)
      : x(std::move(x)), m(m), n(n)
  {}
  El::BigFloat x = 0;
  int m{};
  int n{};

  // Default constructor is needed to make boost::serialization code easier
  Mathematica_Function_F0() = default;
  friend bool operator==(const Mathematica_Function_F0 &lhs,
                         const Mathematica_Function_F0 &rhs)
  {
    return lhs.x == rhs.x && lhs.m == rhs.m && lhs.n == rhs.n;
  }
  friend bool operator!=(const Mathematica_Function_F0 &lhs,
                         const Mathematica_Function_F0 &rhs)
  {
    return !(lhs == rhs);
  }
};

struct Mathematica_Function_F
{
  Mathematica_Function_F(std::string stamp, const int L, const int m,
                         const int n, El::BigFloat x)
      : stamp(std::move(stamp)), L(L), m(m), n(n), x(std::move(x))
  {}
  std::string stamp;
  int L{};
  int m{};
  int n{};
  El::BigFloat x;

  // Default constructor is needed to make boost::serialization code easier
  Mathematica_Function_F() = default;

  friend bool operator==(const Mathematica_Function_F &lhs,
                         const Mathematica_Function_F &rhs)
  {
    return lhs.stamp == rhs.stamp && lhs.L == rhs.L && lhs.m == rhs.m
           && lhs.n == rhs.n && lhs.x == rhs.x;
  }
  friend bool operator!=(const Mathematica_Function_F &lhs,
                         const Mathematica_Function_F &rhs)
  {
    return !(lhs == rhs);
  }
};

struct Mathematica_Function_FS
{
  Mathematica_Function_FS(std::string stamp, const int L, const int m,
                          const int n, El::BigFloat x)
      : stamp(std::move(stamp)), L(L), m(m), n(n), x(std::move(x))
  {}
  std::string stamp;
  int L{};
  int m{};
  int n{};
  El::BigFloat x;

  // Default constructor is needed to make boost::serialization code easier
  Mathematica_Function_FS() = default;

  friend bool operator==(const Mathematica_Function_FS &lhs,
                         const Mathematica_Function_FS &rhs)
  {
    return lhs.stamp == rhs.stamp && lhs.L == rhs.L && lhs.m == rhs.m
           && lhs.n == rhs.n && lhs.x == rhs.x;
  }
  friend bool operator!=(const Mathematica_Function_FS &lhs,
                         const Mathematica_Function_FS &rhs)
  {
    return !(lhs == rhs);
  }
};

// Linear combination of functions F0, F and FS.
// This is oddly specific, but is necessary for the chosen parallelization scheme for objective/normalization:
// Each rank reads only some of the functions F0,F,FS, and sets others to zero.
// Then the final result is obtained via mpi_allreduce.
struct Linear_Combination_Of_Mathematica_Functions
{
  using Mathematica_Function
    = std::variant<Mathematica_Function_F0, Mathematica_Function_F,
                   Mathematica_Function_FS>;

  // Each term is a pair of (coefficient, value), e.g. 1.23 * F0[x,4,5]
  std::vector<std::pair<El::BigFloat, Mathematica_Function>> terms;

  Linear_Combination_Of_Mathematica_Functions() = default;

  Linear_Combination_Of_Mathematica_Functions(El::BigFloat coeff,
                                              Mathematica_Function func)
      : terms({{std::move(coeff), std::move(func)}})
  {}

  explicit Linear_Combination_Of_Mathematica_Functions(
    Mathematica_Function func)
      : Linear_Combination_Of_Mathematica_Functions(1, std::move(func))
  {}

  // Copy and move operations

  Linear_Combination_Of_Mathematica_Functions(
    const Linear_Combination_Of_Mathematica_Functions &other)
    = default;
  Linear_Combination_Of_Mathematica_Functions(
    Linear_Combination_Of_Mathematica_Functions &&other) noexcept
    = default;
  Linear_Combination_Of_Mathematica_Functions &
  operator=(const Linear_Combination_Of_Mathematica_Functions &other)
  {
    if(this == &other)
      return *this;
    terms = other.terms;
    return *this;
  }
  Linear_Combination_Of_Mathematica_Functions &
  operator=(Linear_Combination_Of_Mathematica_Functions &&other) noexcept
  {
    if(this == &other)
      return *this;
    terms = std::move(other.terms);
    return *this;
  }

  friend bool
  operator==(const Linear_Combination_Of_Mathematica_Functions &lhs,
             const Linear_Combination_Of_Mathematica_Functions &rhs)
  {
    return lhs.terms == rhs.terms;
  }
  friend bool
  operator!=(const Linear_Combination_Of_Mathematica_Functions &lhs,
             const Linear_Combination_Of_Mathematica_Functions &rhs)
  {
    return !(lhs == rhs);
  }

  // Arithmetic operations

  Linear_Combination_Of_Mathematica_Functions &
  operator*=(const El::BigFloat &alpha)
  {
    for(auto &[coeff, func] : terms)
      coeff *= alpha;
    return *this;
  }

  Linear_Combination_Of_Mathematica_Functions &
  operator/=(const El::BigFloat &alpha)
  {
    return operator*=(1 / alpha);
  }

  Linear_Combination_Of_Mathematica_Functions &
  operator+=(const Linear_Combination_Of_Mathematica_Functions &other)
  {
    terms.reserve(terms.size() + other.terms.size());
    terms.insert(terms.end(), other.terms.begin(), other.terms.end());
    return *this;
  }

  Linear_Combination_Of_Mathematica_Functions &
  operator-=(const Linear_Combination_Of_Mathematica_Functions &other)
  {
    auto minus_other = other;
    minus_other *= -1;
    return operator+=(minus_other);
  }
};

// Parallel evaluation ofstd::vector<Linear_Combination_Of_Mathematica_Functions>
// (used for objective and normalization)

// Copied from https://en.cppreference.com/w/cpp/utility/variant/visit2
// helper type for the visitor #4
template <class... Ts> struct overloaded : Ts...
{
  using Ts::operator()...;
};
// explicit deduction guide (not needed as of C++20)
template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

// Each rank reads and evaluates only some of the symbols F0, F, FS,
// and sets others to zero.
// After that, the partial results from all ranks are combined via mpi::AllReduce.
template <class TDataProvider>
[[nodiscard]] std::vector<El::BigFloat> evaluate_parallel(
  const std::vector<Linear_Combination_Of_Mathematica_Functions> &input,
  TDataProvider &data_provider,
  const El::mpi::Comm &comm = El::mpi::COMM_WORLD)
{
  std::vector<El::BigFloat> result(input.size(), 0);

  using Function
    = Linear_Combination_Of_Mathematica_Functions::Mathematica_Function;

  // Round-robin distribution of (stamp,L) among ranks
  const auto get_rank = [&](const Function &func) -> int {
    return std::visit(
      [&](auto &&arg) -> int {
        using T = std::decay_t<decltype(arg)>;
        if constexpr(std::is_same_v<T, Mathematica_Function_F0>)
          return 0;
        else
          return data_provider.get_index(arg.stamp, arg.L) % comm.Size();
      },
      func);
  };

  const auto eval = [&](const Function &func) -> El::BigFloat {
    return std::visit(
      overloaded{
        [&](const Mathematica_Function_F0 &arg) {
          return data_provider.F0(arg.x, arg.m, arg.n);
        },
        [&](const Mathematica_Function_F &arg) {
          return data_provider.F(arg.stamp, arg.L, arg.m, arg.n, arg.x);
        },
        [&](const Mathematica_Function_FS &arg) {
          return data_provider.FS(arg.stamp, arg.L, arg.m, arg.n, arg.x);
        }},
      func);
  };

  for(size_t i = 0; i < input.size(); ++i)
    {
      for(const auto &[coeff, func] : input.at(i).terms)
        {
          if(comm.Rank() == get_rank(func))
            result.at(i) += coeff * eval(func);
        }
    }

  // Sum of partial results from all ranks
  El::mpi::AllReduce(result.data(), result.size(), El::mpi::SUM, comm);
  return result;
}

// Serialization

namespace boost::serialization
{
  template <class Archive>
  void
  serialize(Archive &ar, Mathematica_Function_F0 &f0, const version_type &)
  {
    ar & f0.x;
    ar & f0.m;
    ar & f0.n;
  }
  template <class Archive>
  void serialize(Archive &ar, Mathematica_Function_F &f, const version_type &)
  {
    ar & f.stamp;
    ar & f.L;
    ar & f.m;
    ar & f.n;
    ar & f.x;
  }

  template <class Archive>
  void
  serialize(Archive &ar, Mathematica_Function_FS &fs, const version_type &)
  {
    ar & fs.stamp;
    ar & fs.L;
    ar & fs.m;
    ar & fs.n;
    ar & fs.x;
  }

  template <class Archive>
  void serialize(
    Archive &ar,
    Linear_Combination_Of_Mathematica_Functions::Mathematica_Function &func,
    const boost::serialization::version_type &)
  {
    size_t index;
    if(Archive::is_saving::value)
      index = func.index();

    ar & index;

    if(Archive::is_loading::value)
      {
        if(index == 0)
          func.emplace<0>();
        else if(index == 1)
          func.emplace<1>();
        else if(index == 2)
          func.emplace<2>();
        static_assert(
          std::variant_size_v<
            Linear_Combination_Of_Mathematica_Functions::Mathematica_Function>
          == 3);
      }

    std::visit([&](auto &&arg) { ar & arg; }, func);
  }

  // template <class Archive, class T1, class T2>
  // void serialize(Archive &ar, std::pair<T1, T2> &p, const version_type &)
  // - causes compilation errors with Boost 1.68
  template <class Archive>
  void serialize(
    Archive &ar,
    std::pair<El::BigFloat,
              Linear_Combination_Of_Mathematica_Functions::Mathematica_Function>
      &p,
    const version_type &)
  {
    ar & p.first;
    ar & p.second;
  }

  template <class Archive>
  void
  serialize(Archive &ar, Linear_Combination_Of_Mathematica_Functions &funcs,
            const version_type &)
  {
    // TODO use boost::serialization::make_array to make it more compact?
    ar & funcs.terms;
  }
}

// BOOST_SERIALIZATION_SPLIT_FREE(El::Matrix<El::BigFloat>)

// https://www.boost.org/doc/libs/1_82_0/libs/serialization/doc/special.html#objecttracking
// We are just writing arrays, don't need the object tracking mechanism,
// which may cause memory issues (according to some StackOverflow questions).
BOOST_CLASS_TRACKING(Mathematica_Function_F0,
                     boost::serialization::track_never)
BOOST_CLASS_TRACKING(Mathematica_Function_F, boost::serialization::track_never)
BOOST_CLASS_TRACKING(Mathematica_Function_FS,
                     boost::serialization::track_never)
BOOST_CLASS_TRACKING(
  Linear_Combination_Of_Mathematica_Functions::Mathematica_Function,
  boost::serialization::track_never)
BOOST_CLASS_TRACKING(
  decltype(Linear_Combination_Of_Mathematica_Functions::terms)::value_type,
  boost::serialization::track_never)
BOOST_CLASS_TRACKING(Linear_Combination_Of_Mathematica_Functions,
                     boost::serialization::track_never)
