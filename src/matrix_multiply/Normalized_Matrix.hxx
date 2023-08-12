// #pragma once
//
// #include <El.hpp>
//
// struct Normalized_Matrix{
//   enum Normalization_Kind
//   {
//     Columns,
//     Rows,
//     Elements
//   };
//
//   Normalization_Kind normalization_kind;
//   El::Matrix<El::BigFloat> normalized_matrix;
//
//   Normalized_Matrix(const El::Matrix<El::BigFloat> & matrix,
//   Normalization_Kind normalization_kind);
//
//   void restore(El::Matrix<El::BigFloat>& result);
//
//   static void dgemm();
// private:
//   void initialize_norm(const El::Matrix<El::BigFloat> & matrix,
//   Normalization_Kind normalization_kind); El::Matrix<El::BigFloat>
//   norm_elements; std::vector<El::BigFloat> norm_rowcol;
// };