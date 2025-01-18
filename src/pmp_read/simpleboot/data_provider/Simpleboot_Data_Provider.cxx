#include "Simpleboot_Data_Provider.hxx"

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/serialization/vector.hpp>

// TODO: El::BigFloat serialization is already specified "sdpb_util/boost_serialization.hxx".
// Here we use different serialization parameters.
// Ideally, we should use only boost_serialization.hxx.
// NB: this requires to change the code that generates block files.

namespace boost::serialization
{
  template <class Archive>
  void save(Archive &ar, El::BigFloat const &f,
            const boost::serialization::version_type &)
  {
    std::vector<uint8_t> local_array(f.SerializedSize());
    f.Serialize(local_array.data());
    ar & local_array;
  }

  template <class Archive>
  void load(Archive &ar, El::BigFloat &f,
            const boost::serialization::version_type &)
  {
    std::vector<uint8_t> local_array(f.SerializedSize());
    ar & local_array;
    f.Deserialize(local_array.data());
  }
} // namespace boost::serialization

BOOST_SERIALIZATION_SPLIT_FREE(El::BigFloat)

static constexpr auto boost_archive_flags
  = boost::archive::no_header | boost::archive::no_tracking;

namespace fs = std::filesystem;
using block_type = std::vector<std::vector<std::vector<El::BigFloat>>>;

namespace
{
  block_type read_block_file(const fs::path &file)
  {
    std::vector<std::vector<std::vector<El::BigFloat>>> zzb_derivs_conv_El;
    std::ifstream ifs(file);
    ASSERT(ifs.good(), "Failed to open block file: ", file);
    boost::archive::binary_iarchive ia(ifs, boost_archive_flags);
    ia & zzb_derivs_conv_El;
    return zzb_derivs_conv_El;
  }

  std::map<std::pair<std::string, int>, int>
  generate_blockF_key2index(const std::string &block_folder)
  {
    namespace fs = std::filesystem;
    using block_key_type = std::pair<std::string, int>;
    std::map<block_key_type, int> blockF_key2index;

    // Note that directory_iterator does not specify items order.
    // we put keys in map first (thus sorting them) and enumerate at the end.
    for(auto const &file : fs::directory_iterator(block_folder))
      {
        if(fs::is_regular_file(file)
           && file.path().extension() == std::string(".block"))
          {
            const std::string filename = file.path().filename().string();
            size_t barL = filename.find("-L");
            if(barL == std::string::npos)
              RUNTIME_ERROR("Load block error : invalid block file name : ",
                            filename);
            const std::string stamp = filename.substr(0, barL);
            barL += 2;
            const size_t dot = filename.find(".", barL);
            if(dot == std::string::npos)
              RUNTIME_ERROR("Load block error : invalid block file name : ",
                            filename);
            int spin = std::stoi(filename.substr(barL, dot));

            blockF_key2index.emplace(std::make_pair(stamp, spin), 0);
          }
      }

    int i = 0;
    for(auto &[key, index] : blockF_key2index)
      index = i++;
    return blockF_key2index;
  }
}

Simpleboot_Data_Provider::Simpleboot_Data_Provider(
  const Simpleboot_Parameters &params)
    : Abstract_Simpleboot_Data_Provider(params),
      block_folder(params.block_folder),
      blockF_key2index(generate_blockF_key2index(block_folder))
{}

int Simpleboot_Data_Provider::get_index(const std::string &stamp,
                                        int spin) const
{
  return blockF_key2index.at({stamp, spin});
}

Polynomial
Simpleboot_Data_Provider::blockF_lookup(const std::string &stamp, const int L,
                                        const int m, const int n)
{
  const auto &block = get_blockF(stamp, L);
  if(block.size() < m + 1 || block.at(m).size() < n + 1)
    RUNTIME_ERROR("can't find polynomial for stamp=", stamp, " L=", L,
                  " m=", m, " n=", n, DEBUG_STRING(block.size()),
                  DEBUG_STRING(block.at(m).size()));

  Polynomial polynomial;
  polynomial.coefficients = block.at(m).at(n);
  return polynomial;
}

// To prevent extra copying, we return const reference to a block stored in cache.
// Since the reference is used only locally by blockF_lookup, it remains alive.
const Simpleboot_Data_Provider::block_type &
Simpleboot_Data_Provider::get_blockF(const std::string &stamp, const int spin)
{
  namespace fs = std::filesystem;

  const auto key = std::make_pair(stamp, spin);
  {
    const auto it = blockF_cache.find(key);
    if(it != blockF_cache.end())
      return it->second;
  }

  const fs::path file
    = block_folder / build_string(stamp, "-L", spin, ".block");

  const auto [it, res] = blockF_cache.emplace(key, read_block_file(file));
  ASSERT(res, "Failed to add block to cache: ", DEBUG_STRING(stamp),
         DEBUG_STRING(spin));

  return it->second;
}
