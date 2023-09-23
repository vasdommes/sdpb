#pragma once

#include <El.hpp>

#include <boost/noncopyable.hpp>

#include <cassert>
#include <numeric>
#include <vector>

template <class T> class Shared_Window_Array : boost::noncopyable
{
public:
  MPI_Win win;
  El::mpi::Comm comm;
  T *data;
  const size_t size;

public:
  Shared_Window_Array() = delete;
  // shared_memory_comm should be created via
  // MPI_Comm_split_type (MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
  // MPI_INFO_NULL, &shared_memory_comm);
  //
  // It ensures that all ranks in the communicator are on the same node
  // and can share memory.
  Shared_Window_Array(El::mpi::Comm shared_memory_comm, size_t size)
      : comm(shared_memory_comm), size(size)
  {
    MPI_Aint local_window_size; // number of bytes allocated by current rank
    int disp_unit = sizeof(T);

    // Allocate all memory in rank=0
    if(El::mpi::Rank(shared_memory_comm) == 0)
      local_window_size = size * disp_unit;
    else
      local_window_size = 0;

    MPI_Win_allocate_shared(local_window_size, disp_unit, MPI_INFO_NULL,
                            shared_memory_comm.comm, &data, &win);
    // Get local pointer to data allocated in rank=0
    MPI_Win_shared_query(win, 0, &local_window_size, &disp_unit, &data);
    assert(local_window_size == size * sizeof(T));
    assert(disp_unit == sizeof(T));
    Fence();
  }

  ~Shared_Window_Array()
  {
    Fence();
    MPI_Win_free(&win);
  }

  void Fence() { MPI_Win_fence(0, win); }
  T &operator[](size_t index) { return data[index]; }
  const T &operator[](size_t index) const { return data[index]; }
};

// Matrix over shared memory window
template <class T> class Shared_Memory_Matrix : boost::noncopyable
{
private:
  El::Matrix<T> matrix;
  Shared_Window_Array<T> data;

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
// Store matrix of residues (casted to T=double)
// All of them are stored consecutively,
// in prime-block-row-major order
// NB we need single window for all primes, to use FLINT comb
// TODO change to col-major, it might be better for BLAS
// we can use LDim (= sum of heights) to split into horizontal bands
template <class T> class Block_Residue_Matrices_Window_Old : boost::noncopyable
{
private:
  Shared_Window_Array<T> window;
  //  std::vector<size_t> block_offsets;

public:
  const size_t num_primes;
  const size_t num_blocks;
  const std::vector<size_t> block_heights;
  const size_t height;
  const size_t width;

  // residues[prime_index] is a tall matrix containing residues of each block,
  // stacked on top of each other:
  // block_residues[prime_index][0]
  // block_residues[prime_index][1]
  // ...
  // block_residues[prime_index][num_blocks-1]
  //
  // residues[prime_index] is a regular matrix attached to our memory window,
  // and block_residues[prime_index][block_index] are its submatrices
  // (referencing the same data)
  std::vector<El::Matrix<T>> residues;
  // block_residues[prime_index][block_index] = residue of block modulo prime
  std::vector<std::vector<El::Matrix<T>>> block_residues;

  Block_Residue_Matrices_Window_Old() = delete;
  Block_Residue_Matrices_Window_Old(MPI_Comm shared_memory_comm,
                                    size_t num_primes, size_t num_blocks,
                                    const std::vector<size_t> &block_heights,
                                    size_t block_width)
      : num_primes(num_primes),
        num_blocks(num_blocks),
        block_heights(block_heights),
        height(Sum(block_heights)),
        width(block_width),
        window(shared_memory_comm, num_primes * height * width)
  {
    //    block_offsets.resize(num_blocks);
    //    size_t curr_offset = 0;
    //    for(size_t i = 0; i < num_blocks; ++i)
    //      {
    //        block_offsets[i] = curr_offset;
    //        curr_offset += block_heights.at(i) * block_width;
    //      }

    residues.resize(num_primes);
    block_residues.resize(num_primes);
    for(size_t prime_index = 0; prime_index < num_primes; ++prime_index)
      {
        size_t prime_offset = prime_index * PrimeStride();
        residues.at(prime_index)
          .Attach(height, width, window.data + prime_offset);

        block_residues.at(prime_index).resize(num_blocks);
        size_t block_start_row = 0;
        for(size_t block_index = 0; block_index < num_blocks; ++block_index)
          {
            size_t block_height = block_heights.at(block_index);
            El::Range<El::Int> I(block_start_row,
                                 block_start_row + block_height);
            El::Range<El::Int> J(0, width);
            block_residues.at(prime_index).at(block_index)
              = residues.at(prime_index)(I, J);

            block_start_row += block_height;
          }
      }
  }

  //  T *Data(size_t offset = 0)
  //  {
  //    assert(offset < window.size);
  //    return window.data + offset;
  //  }

  //  size_t Offset(size_t prime_index, size_t block_index, size_t i, size_t j)
  //  {
  //    assert(prime_index < num_primes);
  //    assert(block_index < num_blocks);
  //    assert(i < block_heights.at(block_index));
  //    assert(j < width);
  //    size_t result = prime_index * PrimeStride() +
  //    block_offsets.at(block_index)
  //                    + i * width + j;
  //    assert(result < window.size);
  //    return result;
  //  }

  //  [[nodiscard]] T
  //  Get(size_t prime_index, size_t block_index, size_t i, size_t j) const
  //  {
  //    //    return window[Offset(prime_index, block_index, i, j)];
  //    return block_residues.at(prime_index).at(block_index).Get(i, j);
  //  }
  //
  //  void Set(size_t prime_index, size_t block_index, size_t i, size_t j, T
  //  value)
  //  {
  //    //    window[Offset(prime_index, block_index, i, j)] = value;
  //    block_residues.at(prime_index).at(block_index).Set(i, j, value);
  //  }

  [[nodiscard]] size_t PrimeStride() const { return width * height; }
  [[nodiscard]] size_t ColStride() const { return height; }

  void Fence() { window.Fence(); }

private:
  static size_t Sum(const std::vector<size_t> &block_heights)
  {
    return std::accumulate(block_heights.begin(), block_heights.end(), 0);
  }
};

// Vector of matrices stored in a contiguous Shared_Window_Array
// in prime-row-major order
// This is residues of Q (BLAS multiplication output is written to there)
// NB: El::Matrix is col-major, so we can't reuse it here
template <class T> class Residue_Matrices_Window : boost::noncopyable
{
public:
  const size_t num_primes;
  const size_t height;
  const size_t width;
  const size_t prime_stride;
  std::vector<El::Matrix<T>> residues;

private:
  Shared_Window_Array<T> window;

public:
  Residue_Matrices_Window(El::mpi::Comm shared_memory_comm, size_t num_primes,
                          size_t height, size_t width)
      : num_primes(num_primes),
        height(height),
        width(width),
        prime_stride(height * width),
        window(shared_memory_comm, num_primes * prime_stride)
  {
    assert(num_primes > 0);
    assert(height > 0);
    assert(width > 0);
    residues.resize(num_primes);
    for(size_t prime_index = 0; prime_index < num_primes; ++prime_index)
      {
        size_t prime_offset = prime_index * prime_stride;
        residues.at(prime_index)
          .Attach(height, width, window.data + prime_offset, height);
      }
  }
  El::mpi::Comm Comm() const { return window.comm; }
  void Fence() { window.Fence(); }
};

// Same as Residue_Matrices_Window<T>,
// but each (tall) residue matrix (i.e. residues[prime_index])
// is split horizontally into blocks
// (i.e. block_residues[prime_index][0..num_blocks-1])
template <class T>
class Block_Residue_Matrices_Window : public Residue_Matrices_Window<T>
{
public:
  const size_t num_blocks;

  // block_residues[prime_index][block_index] = residue of block modulo prime
  // These matrices are views over residues[prime_index],
  // which is a tall matrix containing residues of each block,
  // stacked on top of each other:
  // block_residues[prime_index][0]
  // block_residues[prime_index][1]
  // ...
  // block_residues[prime_index][num_blocks-1]
  //
  // residues[prime_index] is a regular matrix attached to our memory window,
  // and block_residues[prime_index][block_index] is a view to its submatrix
  std::vector<std::vector<El::Matrix<T>>> block_residues;

  Block_Residue_Matrices_Window(El::mpi::Comm shared_memory_comm,
                                size_t num_primes, size_t num_blocks,
                                const std::vector<El::Int> &block_heights,
                                size_t block_width)
      : Residue_Matrices_Window<T>(shared_memory_comm, num_primes,
                                   Sum(block_heights), block_width),
        num_blocks(num_blocks),
        block_residues(num_primes, std::vector<El::Matrix<T>>(num_blocks))
  {
    for(size_t prime_index = 0; prime_index < num_primes; ++prime_index)
      {
        size_t block_start_row = 0;
        for(size_t block_index = 0; block_index < num_blocks; ++block_index)
          {
            size_t block_height = block_heights.at(block_index);
            El::Range<El::Int> I(block_start_row,
                                 block_start_row + block_height);
            El::Range<El::Int> J(0, this->width);
            El::View(block_residues.at(prime_index).at(block_index),
                     this->residues.at(prime_index), I, J);
            block_start_row += block_height;
          }
      }
    if(num_primes > 0 && num_blocks > 0)
      assert(block_residues[0][0].Buffer() == this->residues[0].Buffer());
  }

private:
  static El::Int Sum(const std::vector<El::Int> &block_heights)
  {
    return std::accumulate(block_heights.begin(), block_heights.end(), 0);
  }
};
