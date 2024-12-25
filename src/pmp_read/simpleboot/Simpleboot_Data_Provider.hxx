#pragma once

#include "Abstract_Simpleboot_Data_Provider.hxx"
#include "mathematica_parse_util.hxx"
#include "sdpb_util/Boost_Float.hxx"

#include <filesystem>

class Simpleboot_Data_Provider : public Abstract_Simpleboot_Data_Provider
{
public:
  explicit Simpleboot_Data_Provider(const Simpleboot_Parameters &parameters);

protected:
  Polynomial
  blockF_lookup(const std::string &stamp, int L, int m, int n) override;

private:
  std::filesystem::path block_folder;
  std::map<std::pair<std::string, int>,
           std::vector<std::vector<std::vector<El::BigFloat>>>>
    blockF;

private:
  void load_block_folder();
  // TODO rename?
  void load_block_folder(const std::string &stamp, int spin);
};