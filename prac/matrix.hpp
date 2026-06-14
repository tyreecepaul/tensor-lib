#ifndef __MATRIX_H
#define __MATRIX_H

#include <vector>

template <typename T> class Matrix {
private:
  std::vector<std::vector<T>> mat;
  unsigned rows;
  unsigned cols;

public:
  Matrix(unsigned _rows, unsigned _cols, const T &_initial);
  Matrix(const Matrix<T> &rhs);
  virtual ~Matrix();

  // operator overloading for "standard" mathematical matrix operations
  Matrix<T> &operator=(const Matrix<T> &rhs);

  // matrix mathematical operations
  Matrix<T> operator+(const Matrix<T> &rhs);
  Matrix<T> &operator+=(const Matrix<T> &rhs);
  Matrix<T> operator-(const Matrix<T> &rhs);
  Matrix<T> &operator-=(const Matrix<T> &rhs);
  Matrix<T> operator*(const Matrix<T> &rhs);
  Matrix<T> &operator*=(const Matrix<T> &rhs);
  Matrix<T> transpose();

  // matrix/scalar operations
  Matrix<T> operator+(const T &rhs);
  Matrix<T> operator-(const T &rhs);
  Matrix<T> operator*(const T &rhs);
  Matrix<T> operator/(const T &rhs);

  // matrix/vector operations
  std::vector<T> operator*(const std::vector<T> &rhs);
  std::vector<T> diag_vec();

  // accessing individual elements
  T &operator()(const unsigned &row, const unsigned &col);
  const T &operator()(const unsigned &row, const unsigned &col) const;

  // access row and col size
  unsigned get_rows() const;
  unsigned get_cols() const;
};

#include "matrix.tpp"

#endif
