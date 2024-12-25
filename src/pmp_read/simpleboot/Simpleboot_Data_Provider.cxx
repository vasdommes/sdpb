#include "Simpleboot_Data_Provider.hxx"

#include <boost/algorithm/string.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/math/tools/polynomial.hpp>
#include <boost/serialization/vector.hpp>

namespace boost
{
  namespace serialization
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

  } // namespace serialization
} // namespace boost

BOOST_SERIALIZATION_SPLIT_FREE(El::BigFloat)

static auto const boost_archive_flags
  = boost::archive::no_header | boost::archive::no_tracking;

Simpleboot_Data_Provider::Simpleboot_Data_Provider(
  const Simpleboot_Parameters &parameters)
    : Abstract_Simpleboot_Data_Provider(parameters)
{
  load_block_folder();
}

Polynomial Simpleboot_Data_Provider::blockF_lookup(const std::string &stamp,
                                                   int L, int m, int n)
{
  auto pblock = blockF.find(std::make_pair(stamp, L));
  if(pblock == blockF.end())
    {
      load_block_folder(stamp, L);
      pblock = blockF.find(std::make_pair(stamp, L));
    }

  if(pblock->second.size() < m + 1 || pblock->second.at(m).size() < n + 1)
    RUNTIME_ERROR("can't find polynomial for stamp=", stamp, " L=", L,
                  " m=", m, " n=", n);

  Polynomial polynomial;
  polynomial.coefficients = pblock->second.at(m).at(n);
  return polynomial;
}
void Simpleboot_Data_Provider::load_block_folder()
{
  namespace fs = std::filesystem;

  El::Output("scan blocks...");

  std::vector<std::vector<std::vector<El::BigFloat>>> zzb_derivs_conv_El;

  for(auto const &file : fs::recursive_directory_iterator(block_folder))
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
          size_t dot = filename.find(".", barL);
          if(dot == std::string::npos)
            RUNTIME_ERROR("Load block error : invalid block file name : ",
                          filename);
          int spin = std::stoi(filename.substr(barL, dot));

          El::Output("load block with stamp = ", stamp, " spin = ", spin);

          std::ifstream ifs(file.path());
          boost::archive::binary_iarchive ia(ifs, boost_archive_flags);
          ia & zzb_derivs_conv_El;

          blockF.emplace(std::make_pair(stamp, spin), zzb_derivs_conv_El);
        }
    }

  El::Output("blocks loaded");

  // for(const auto &[key, value] : blockF)
  //   El::Output("find block with stamp=", key.first, " spin=", key.second,
  //              " max_m=", value.size() - 1);
}

void Simpleboot_Data_Provider::load_block_folder(const std::string &stamp,
                                                 int spin)
{
  namespace fs = std::filesystem;

  std::vector<std::vector<std::vector<El::BigFloat>>> zzb_derivs_conv_El;

  auto pblock = blockF.find(std::make_pair(stamp, spin));
  if(pblock != blockF.end())
    RUNTIME_ERROR("load_block_folder error : ", stamp, "-L", spin, ".block",
                  " already exist.");

  std::stringstream path_str;
  path_str << block_folder << "/" << stamp << "-L" << spin << ".block";

  const fs::path file(path_str.str());

  if(!fs::is_regular_file(file)) //exists(file) &&
    RUNTIME_ERROR("load_block_folder error : ", stamp, "-L", spin, ".block",
                  " is missing.");

  std::ifstream ifs(file);
  boost::archive::binary_iarchive ia(ifs, boost_archive_flags);
  ia & zzb_derivs_conv_El;

  blockF.emplace(std::make_pair(stamp, spin), zzb_derivs_conv_El);

  El::Output("Rank=", El::mpi::Rank(), " : load block with stamp = ", stamp,
             " spin = ", spin);
}
