#include "sdpb_util/assert.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "spectrum/Zeros.hxx"
#include "sdpb_util/ostream/set_stream_precision.hxx"

#include <rapidjson/ostreamwrapper.h>
#include <rapidjson/prettywriter.h>

#include <filesystem>

namespace fs = std::filesystem;

// Helper class to simplify writing JSON
// TBaseWriter is rapidjson::Writer<...> or rapidjson::PrettyWriter<...>
// TODO move to sdpb_util, reuse for write_pmp_info and save_c_minus_By
template <class TBaseWriter> class Json_BigFloat_Writer : public TBaseWriter
{
public:
  template <class... TArgs>
  explicit Json_BigFloat_Writer(TArgs &&...args)
      : TBaseWriter(std::forward<TArgs>(args)...)
  { set_stream_precision(ss); }

  auto BigFloat(const El::BigFloat &value)
  {
    ss.str({});
    ss << value;
    return this->String(ss.str().c_str());
  }

private:
  // Reusable stream for writing BigFloats
  std::stringstream ss;
};

using Json_Writer
  = Json_BigFloat_Writer<rapidjson::Writer<rapidjson::OStreamWrapper>>;
using Json_PrettyWriter
  = Json_BigFloat_Writer<rapidjson::PrettyWriter<rapidjson::OStreamWrapper>>;

void write_file(const fs::path &output_path,
                const std::vector<Zeros> &zeros_blocks, Timers &timers)
{
  if(El::mpi::Rank() != 0)
    return;

  Scoped_Timer timer(timers, "write_file");

  if(output_path.has_parent_path())
    fs::create_directories(output_path.parent_path());
  std::ofstream ofs(output_path);
  ASSERT(ofs.good(), "Problem when opening output file: ", output_path);

  rapidjson::OStreamWrapper ows(ofs);
  Json_PrettyWriter writer(ows);
  writer.SetIndent(' ', 2);

  writer.StartArray();
  for(auto zeros_iterator(zeros_blocks.begin());
      zeros_iterator != zeros_blocks.end(); ++zeros_iterator)
    {
      const auto &zeros = *zeros_iterator;
      const auto block_path = zeros_iterator->block_path.string();
      // TODO store block_index in Zeros?
      ASSERT(!block_path.empty(), "Empty path for block_",
             std::distance(zeros_blocks.begin(), zeros_iterator));
      writer.StartObject();
      {
        writer.Key("block_path");
        writer.String(block_path.c_str());

        writer.Key("zeros");
        writer.StartArray();
        for(const auto &zero : zeros.zeros)
          {
            writer.StartObject();
            {
              writer.Key("zero");
              writer.BigFloat(zero.zero);

              writer.Key("lambda");
              const auto &lambda = zero.lambda;
              if(lambda.Height() > 0)
                {
                  ASSERT_EQUAL(lambda.Width(), 1,
                               "lambda should contain a single eigenvector!");
                }
              writer.StartArray();
              for(int row = 0; row < lambda.Height(); ++row)
                writer.BigFloat(lambda(row, 0));
              writer.EndArray();
            }
            writer.EndObject();
          }
        writer.EndArray();

        writer.Key("error");
        writer.BigFloat(zeros.error);
      }
      writer.EndObject();
    }
  writer.EndArray();
  ASSERT(writer.IsComplete());
  ASSERT(ofs.good(), "Problem when writing to output file: ", output_path);
}
