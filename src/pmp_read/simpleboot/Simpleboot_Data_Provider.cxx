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
}

Simpleboot_Data_Provider::Simpleboot_Data_Provider(
  const Simpleboot_Parameters &parameters)
    : Abstract_Simpleboot_Data_Provider(parameters)
{}

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
