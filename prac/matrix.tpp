// parameter constructor
template <typename T>
Matrix<T>::Matrix(unsigned _rows, unsigned _cols, const T &_initial) {
  mat.resize(_rows);
  for (unsigned i = 0; i < mat.size(); i++) {
    mat[i].resize(_cols, _initial);
  }
  rows = _rows;
  cols = _cols;
}

// copy constructor
template <typename T> Matrix<T>::Matrix(const Matrix<T> &rhs) {
  mat = rhs.mat;
  rows = rhs.get_rows();
  cols = rhs.get_cols();
}

// (virtual) deconstructor
template <typename T> Matrix<T>::~Matrix() {}

// assignment operator
template <typename T> Matrix<T> &Matrix<T>::operator=(const Matrix<T> &rhs) {
  if (&rhs == this)
    return *this;

  unsigned new_rows = rhs.get_rows();
  unsigned new_cols = rhs.get_cols();

  mat.resize(new_rows);
  for (unsigned i = 0; i < mat.size(); i++) {
    mat[i].resize(new_cols);
  }

  for (unsigned i = 0; i < new_rows; i++) {
    for (unsigned j = 0; j < new_cols; j++) {
      mat[i][j] = rhs(i, j);
    }
  }

  rows = new_rows;
  cols = new_cols;

  return *this;
}

// addition of two matrices
template <typename T> Matrix<T> Matrix<T>::operator+(const Matrix<T> &rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] + rhs(i, j);
    }
  }

  return result;
}

// cumulative addition of two matrices
template <typename T> Matrix<T> &Matrix<T>::operator+=(const Matrix<T> &rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      this->mat[i][j] += rhs(i, j);
    }
  }

  return *this;
}

// subtraction of two matrices
template <typename T> Matrix<T> Matrix<T>::operator-(const Matrix<T> &rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] - rhs(i, j);
    }
  }

  return result;
}

// cumulative subtraction of two matrices
template <typename T> Matrix<T> &Matrix<T>::operator-=(const Matrix<T> &rhs) {
  unsigned rows = rhs.get_rows();
  unsigned cols = rhs.get_cols();

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      this->mat[i][j] -= rhs(i, j);
    }
  }

  return *this;
}

// left multiplication of two matrices
template <typename T> Matrix<T> Matrix<T>::operator*(const Matrix<T> &rhs) {
  Matrix<T> result(this->rows, rhs.get_cols(), 0);

  for (unsigned i = 0; i < this->rows; i++) {
    for (unsigned j = 0; j < rhs.get_cols(); j++) {
      for (unsigned k = 0; k < this->cols; k++) {
        result(i, j) += this->mat[i][k] * rhs(k, j);
      }
    }
  }

  return result;
}

// cumulative left multiplication of two matrices
template <typename T> Matrix<T> &Matrix<T>::operator*=(const Matrix<T> &rhs) {
  Matrix result = (*this) * rhs;
  (*this) = result;
  return *this;
}

// calculate transpose of matrix
template <typename T> Matrix<T> Matrix<T>::transpose() {
  Matrix<T> result(cols, rows, 0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(j, i) = this->mat[i][j];
    }
  }

  return result;
}

// matrix/scalar addition
template <typename T> Matrix<T> Matrix<T>::operator+(const T &rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] + rhs;
    }
  }

  return result;
}

// matrix/scalar subtraction
template <typename T> Matrix<T> Matrix<T>::operator-(const T &rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] - rhs;
    }
  }

  return result;
}

// matrix/scalar multiplication
template <typename T> Matrix<T> Matrix<T>::operator*(const T &rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] * rhs;
    }
  }

  return result;
}

// matrix/scalar division
template <typename T> Matrix<T> Matrix<T>::operator/(const T &rhs) {
  Matrix result(rows, cols, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result(i, j) = this->mat[i][j] / rhs;
    }
  }

  return result;
}

// multiply matrix with vector
template <typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T> &rhs) {
  std::vector<T> result(rhs.size(), 0.0);

  for (unsigned i = 0; i < rows; i++) {
    for (unsigned j = 0; j < cols; j++) {
      result[i] += this->mat[i][j] * rhs[j];
    }
  }

  return result;
}

// get vector of diagonal elements
template <typename T> std::vector<T> Matrix<T>::diag_vec() {
  std::vector<T> result(rows, 0.0);

  for (unsigned i = 0; i < rows; i++) {
    result[i] = this->mat[i][i];
  }

  return result;
}

// get individual elements
template <typename T>
T &Matrix<T>::operator()(const unsigned &row, const unsigned &col) {
  return this->mat[row][col];
}

// get individual elements (const)
template <typename T>
const T &Matrix<T>::operator()(const unsigned &row, const unsigned &col) const {
  return this->mat[row][col];
}

// get number of rows in matrix
template <typename T> unsigned Matrix<T>::get_rows() const {
  return this->rows;
}

// get number of cols in matrix
template <typename T> unsigned Matrix<T>::get_cols() const {
  return this->cols;
}

