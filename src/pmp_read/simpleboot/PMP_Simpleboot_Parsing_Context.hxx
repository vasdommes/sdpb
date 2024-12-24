#pragma once

#include "Json_Simpleboot_Float_Parser.hxx"
#include "Json_Simpleboot_Polynomial_Parser.hxx"

template <class TFloat> class Json_Simpleboot_Float_Parser;
class Json_Simpleboot_Polynomial_Parser;

struct PMP_Simpleboot_Parsing_Context
{
  template <class TFloat>
  using Float_Parser = Json_Simpleboot_Float_Parser<TFloat>;
  using Polynomial_Parser = Json_Simpleboot_Polynomial_Parser;
};
