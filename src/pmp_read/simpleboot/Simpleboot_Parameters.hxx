#pragma once

#include "sdpb_util/Boost_Float.hxx"

#include <filesystem>
#include <map>

struct Simpleboot_Parameters final
{
  El::BigFloat dim = -1;
  El::BigFloat nu = -1;
  int kappa = -1;
  int maxderivs = -1; // only used for interval positivity
  std::map<std::string, El::BigFloat> var_map;

  // TODO this is constant, should we move it elsewhere?
  Boost_Float r_crossing_4 = -1;

  std::filesystem::path block_folder;
  std::vector<std::filesystem::path> input_files;
};
