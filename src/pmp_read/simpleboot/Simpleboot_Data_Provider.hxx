#pragma once

#include "Abstract_Simpleboot_Data_Provider.hxx"
#include "mathematica_parse_util.hxx"

#include <filesystem>

class Simpleboot_Data_Provider : public Abstract_Simpleboot_Data_Provider
{
public:
  explicit Simpleboot_Data_Provider(const Simpleboot_Parameters &parameters);

protected:
  Polynomial
  blockF_lookup(const std::string &stamp, int L, int m, int n) override;

private:
  // (stamp, spin)
  using block_key_type = std::pair<std::string, int>;
  using block_type = std::vector<std::vector<std::vector<El::BigFloat>>>;

  std::filesystem::path block_folder;
  std::map<block_key_type, block_type> blockF_cache;

private:
  const block_type &get_blockF(const std::string &stamp, int spin);
};