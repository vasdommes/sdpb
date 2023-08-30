#pragma once

#include <El.hpp>

#include <boost/noncopyable.hpp>

#include <cassert>
#include <numeric>
#include <vector>

template <class T> class Shared_Window_Array : boost::noncopyable
{
public:
  T *data;
  const size_t size;

private:
  MPI_Win win;

public:
  Shared_Window_Array() = delete;
  // shared_memory_comm should be created via
  // MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
  // MPI_INFO_NULL, &shared_memory_comm);
  //
  // It ensures that all ranks in the communicator are on the same node
  // and can share memory.
  Shared_Window_Array(MPI_Comm shared_memory_comm, size_t size) : size(size)
  {
    MPI_Aint local_window_size; // number of bytes allocated by current rank
    int disp_unit = sizeof(T);

    // Allocate all memory in rank=0
    if(El::mpi::Rank(shared_memory_comm) == 0)
      local_window_size = size * disp_unit;
    else
      local_window_size = 0;

    MPI_Win_allocate_shared(local_window_size, disp_unit, MPI_INFO_NULL,
                            shared_memory_comm, &data, &win);
    // Get local pointer to data allocated in rank=0
    MPI_Win_shared_query(win, 0, &local_window_size, &disp_unit, &data);
    assert(local_window_size == size * sizeof(T));
    assert(disp_unit == sizeof(T));
    MPI_Win_fence(0, win);
  }

  ~Shared_Window_Array()
  {
    MPI_Win_fence(0, win);
    MPI_Win_free(&win);
  }

  T &operator[](size_t index) { return data[index]; }
  const T &operator[](size_t index) const { return data[index]; }
};

// Matrix over shared memory window
template <class T> class Shared_Memory_Matrix : boost::noncopyable
{
private:
  El::Matrix<T> matrix;
  Shared_Window_Array<double> data;

public:
  Shared_Memory_Matrix() = delete;
  // shared_memory_comm should be created via
  // MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
  // MPI_INFO_NULL, &shared_memory_comm);
  //
  // It ensures that all ranks in the communicator are on the same node
  // and can share memory.
  Shared_Memory_Matrix(MPI_Comm shared_memory_comm, int height, int width)
      : data(shared_memory_comm, height * width),
        matrix(height, width, data.data, height)
  {}

  T *Buffer() { return data; }
  size_t BufferSize() { return matrix.Height() * matrix.Width(); }
  T Get(int i, int j) const { return matrix.Get(i, j); }
  void Set(int i, int j, const T &value) { return matrix.Set(i, j, value); }
};

// Shared memory window for storing block residues.
// For each prime:
// For each block:
// Store matrix of residues (casted to double)
// All of them are stored consecutively,
// in prime-block-col-major order
// NB we need single window for all primes, to use FLINT comb
// TODO: we can avoid it, if we
template <class T> class Block_Residue_Matrices_Window : boost::noncopyable
{
private:
  Shared_Window_Array<T> data;
  std::vector<size_t> accumulated_heights;

public:
  const size_t num_primes;
  const size_t num_blocks;
  const std::vector<size_t> block_heights;
  const size_t block_width;
  // TODO
  std::vector<std::vector<El::Matrix<T>>> blocks_residues;

  Block_Residue_Matrices_Window() = delete;
  Block_Residue_Matrices_Window(MPI_Comm shared_memory_comm, size_t num_primes,
                                size_t num_blocks,
                                const std::vector<size_t> &block_heights,
                                size_t block_width)
      : data(shared_memory_comm,
             num_primes * TotalResidueHeight(block_heights) * block_width),
        num_primes(num_primes),
        num_blocks(num_blocks),
        block_heights(block_heights),
        block_width(block_width)
  {
    accumulated_heights.resize(block_heights.size());
    std::partial_sum(block_heights.begin(), block_heights.end(),
                     accumulated_heights.begin());

    // TODO do we need it?
    blocks_residues.resize(num_primes);
    size_t block_offset = 0;
    for(size_t prime = 0; prime < num_primes; ++prime)
      {
        blocks_residues.at(prime).resize(num_blocks);
        for(size_t block = 0; block < num_blocks; ++block)
          {
            El::Matrix<T> &matrix = blocks_residues.at(prime).at(block);
            auto height = block_heights.at(block);
            auto width = block_width;
            auto leading_dimension = height; // El::Matrix uses col-major order
            matrix.Attach(height, width, data.data + block_offset,
                          leading_dimension);
            block_offset += height * width;
          }
      }
  }
  size_t GetIndex(size_t prime_index, size_t block_index, size_t i, size_t j)
  {
    auto prev_heights
      = block_index == 0 ? 0 : accumulated_heights[block_index - 1];
    auto total_height = accumulated_heights[num_blocks - 1];
    return prime_index * total_height * block_width
           + block_index * prev_heights * block_width + i + j * block_width;
  }
  double Get(size_t prime_index, size_t block_index, size_t i, size_t j)
  {
    return blocks_residues.at(prime_index).at(block_index).Get(i, j);
    //        return data[GetIndex(prime_index, block_index, i, j)];
  }
  void
  Set(size_t prime_index, size_t block_index, size_t i, size_t j, double value)
  {
    blocks_residues.at(prime_index).at(block_index).Set(i, j, value);
    //    data[GetIndex(prime_index, block_index, i, j)] = value;
  }

  [[nodiscard]] size_t PrimeStride() const
  {
    return block_width * accumulated_heights[num_blocks - 1];
  }

private:
  static size_t TotalResidueHeight(const std::vector<size_t> &block_heights)
  {
    return std::accumulate(block_heights.begin(), block_heights.end(), 0);
  }
};

// Vector of matrices stored in a contiguous Shared_Window_Array
// This is residues of Q (BLAS multiplication output is written to there)
template <class T> class Residue_Matrices_Window : boost::noncopyable
{
private:
  Shared_Window_Array<T> data;

public:
  std::vector<El::Matrix<T>> matrices;
  const size_t prime_stride;
  const size_t primes_size;
  const size_t height;
  const size_t width;

  Residue_Matrices_Window(MPI_Comm shared_memory_comm, size_t primes_size,
                          size_t height, size_t width)
      : data(shared_memory_comm, primes_size * height * width),
        matrices(primes_size),
        prime_stride(height * width),
        primes_size(primes_size),
        height(height),
        width(width)
  {
    assert(primes_size > 0);
    assert(height > 0);
    assert(width > 0);
    for(size_t i = 0; i < primes_size; ++i)
      {
        size_t leading_dimension = height; // El::Matrix uses col-major order
        matrices.at(i).Attach(height, width, data.data + i * prime_stride,
                              leading_dimension);
      }
  }
  double Get(size_t prime_index, size_t i, size_t j)
  {
    return matrices[prime_index].Get(i, j);
  }
  void Set(size_t prime_index, size_t i, size_t j, double value)
  {
    matrices[prime_index].Set(i, j, value);
  }
};