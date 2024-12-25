#pragma once

#include "Mathematica_Parser.hxx"
#include "mathematica_parse_util.hxx"
#include "pmp_read/read_mathematica/parse_SDP/parse_number.hxx"
#include "sdpb_util/assert.hxx"

#include <memory>

template <class TSimpleboot_Data_Provider>
class Mathematica_Simpleboot_Expression_Parser final : public Mathematica_Parser
{
private:
  const char *ptr_MMA_begin = nullptr;
  const char *ptr_MMA_current = nullptr;
  const char *ptr_MMA_end = nullptr;
  const std::shared_ptr<TSimpleboot_Data_Provider> data_provider;

public:
  explicit Mathematica_Simpleboot_Expression_Parser(
    std::shared_ptr<TSimpleboot_Data_Provider> data_provider)
      : data_provider(std::move(data_provider))
  {}


  // for our purpose now, we only need
  // P[stamp, L, m, n, shift] : block derivative polynomial
  // F[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor
  // Fs[stamp, L, m, n, x, dim, kappa] : block derivative value with prefactor (shortened pole)
  const char *parse_MMA_function(const std::string &name, const char *begin,
                                 const char *end, MMA_ELEMENT &result) override
  {
    const char *pstr = begin;
    if(name == "F" || name == "FS")
      {
        MMA_TOKEN token;
        MMA_ELEMENT element;

        int L, m, n;
        El::BigFloat dim, x;

        pstr = parse_get_token(pstr, end,
                               token); // F[stamp, L, m, n, x, dim, kappa]
        std::string stamp(std::move(AS_MMA_TOKEN(token, String)));
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, L);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, m);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, n);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_expr_as_number(pstr, end, x);
        pstr = parse_MMA_check_op(pstr, end, token, ']');

        if(name == "F")
          data_provider->F(stamp, L, m, n, x, result);
        else
          data_provider->FS(stamp, L, m, n, x, result);

        return pstr;
      }

    if(name == "P")
      {
        MMA_TOKEN token;
        MMA_ELEMENT element;

        int L, m, n;
        El::BigFloat x;

        pstr = parse_get_token(pstr, end, token); // P[stamp, L, m, n, shift]
        std::string stamp(std::move(AS_MMA_TOKEN(token, String)));
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, L);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, m);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, n);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_expr_as_number(pstr, end, x);
        pstr = parse_MMA_check_op(pstr, end, token, ']');

        data_provider->P(stamp, L, m, n, x, result);

        return pstr;
      }

    if(name == "PT")
      {
        MMA_TOKEN token;
        MMA_ELEMENT element;

        int L, m, n;
        El::BigFloat a, b;

        pstr = parse_get_token(pstr, end, token); // PT[stamp, L, m, n, a, b]
        std::string stamp(std::move(AS_MMA_TOKEN(token, String)));
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, L);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, m);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, n);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_expr_as_number(pstr, end, a);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_expr_as_number(pstr, end, b);
        pstr = parse_MMA_check_op(pstr, end, token, ']');

        data_provider->PT(stamp, L, m, n, a, b, result);

        return pstr;
      }

    if(name == "F0") // F0[x, m, n]
      {
        MMA_TOKEN token;
        MMA_ELEMENT element;

        int m, n;
        El::BigFloat x;

        pstr = parse_MMA_expr_as_number(pstr, end, x);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, m);
        pstr = parse_MMA_check_op(pstr, end, token, ',');

        pstr = parse_MMA_token_as_int(pstr, end, token, n);
        pstr = parse_MMA_check_op(pstr, end, token, ']');

        data_provider->F0(x, m, n, result);

        return pstr;
      }

    MMA_PARSER_ERROR("parse_MMA_function error : unsupported ",
                     DEBUG_STRING(name), " \n");
  }

  void parse_MMA_symbol(const std::string &name, MMA_ELEMENT &result) override
  {
    auto pvar = data_provider->var_map.find(name);
    if(pvar == data_provider->var_map.end())
      MMA_PARSER_ERROR("can't find symbol ", name);

    SET_MMA_ELEMENT(result, pvar->second, Expression);
  }
};
