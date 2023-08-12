#include "Shared_Window_Array.hxx"

namespace
{
  size_t data_size(size_t primes_size, size_t block_width,
                   const std::vector<size_t> &block_heights)
  {
    // Number of elements in all matrices
    return primes_size * block_width
           * std::accumulate(block_heights.begin(), block_heights.end(), 0);
  }
}

Block_Residue_Matrices_Window::Block_Residue_Matrices_Window(
  MPI_Comm shared_memory_comm, size_t primes_size, size_t blocks_size,
  const std::vector<size_t> &block_heights, size_t block_width)
    : data(shared_memory_comm,
           data_size(primes_size, block_width, block_heights)),
      primes_size(primes_size),
      blocks_size(blocks_size),
      block_heights(block_heights),
      block_width(block_width)
{
  accumulated_heights.resize(block_heights.size());
  std::partial_sum(block_heights.begin(), block_heights.end(),
                   accumulated_heights.begin());
}

// Prime-block-col-major order
size_t
Block_Residue_Matrices_Window::GetIndex(size_t prime_index, size_t block_index,
                                        size_t i, size_t j)
{
  auto prev_heights
    = block_index == 0 ? 0 : accumulated_heights[block_index - 1];
  auto total_height = accumulated_heights[block_index];
  return prime_index * total_height * block_width
         + block_index * prev_heights * block_width + i + j * block_width;
}
double
Block_Residue_Matrices_Window::Get(size_t prime_index, size_t block_index,
                                   size_t i, size_t j)
{
  return data[GetIndex(prime_index, block_index, i, j)];
}
void Block_Residue_Matrices_Window::Set(size_t prime_index, size_t block_index,
                                        size_t i, size_t j, double value)
{
  data[GetIndex(prime_index, block_index, i, j)] = value;
}
