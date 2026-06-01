#include "tensor.hpp"
#include <stdexcept>
// #include <sstream>

// ---------------- constructors ----------------

Tensor::Tensor(float scalar) : _data{scalar}, _shape{}, _stride{} {}

Tensor::Tensor(std::vector<float> data)
    : _data(std::move(data)), _shape{_data.size()}, _stride{1} {}

Tensor::Tensor(std::vector<std::vector<float>> data) {
  std::size_t rows = data.size();
  std::size_t cols = data[0].size();

  for (auto &row : data) {
    if (row.size() != cols)
      throw std::invalid_argument("Non-rectangular matrix");
  }

  _shape = {rows, cols};
  _stride = {cols, 1};

  _data.reserve(rows * cols);

  for (std::size_t i = 0; i < rows; i++) {
    for (std::size_t j = 0; j < cols; j++) {
      _data.push_back(data[i][j]);
    }
  }
}

// ---------------- indexing ----------------

float &Tensor::operator()(std::size_t i) { return _data[i]; }

const float &Tensor::operator()(std::size_t i) const { return _data[i]; }

float &Tensor::operator()(std::size_t i, std::size_t j) {
  return _data[i * _stride[0] + j];
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const {
  return _data[i * _stride[0] + j];
}

// ---------------- shape ----------------

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }

// ---------------- addition ----------------

Tensor Tensor::operator+(const Tensor &other) const {
  if (_shape != other._shape)
    throw std::invalid_argument("shape mismatch");

  Tensor result;
  result._shape = _shape;
  result._stride = _stride;
  result._data.resize(_data.size());

  for (std::size_t i = 0; i < _data.size(); i++) {
    result._data[i] = _data[i] + other._data[i];
  }

  return result;
}

// ---------------- matrix multiply ----------------

Tensor Tensor::operator*(const Tensor &other) const {
  size_t m = _shape[0];
  size_t K = _shape[1];
  size_t n = other._shape[1];

  Tensor result;
  result._shape = {m, n};
  result._stride = {n, 1};
  result._data.assign(m * n, 0.0f);

  const float *A = _data.data();
  const float *B = other._data.data();
  float *C = result._data.data();

  for (size_t i = 0; i < m; i++) {
    for (size_t k = 0; k < K; k++) {
      float a = A[i * K + k];

      size_t bk = k * n;
      size_t ci = i * n;

      for (size_t j = 0; j < n; j++) {
        C[ci + j] += a * B[bk + j];
      }
    }
  }

  return result;
}

// ---------------- printing ----------------

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  if (t._shape.empty()) {
    os << t._data[0];
    return os;
  }

  if (t._shape.size() == 1) {
    os << "[";
    for (std::size_t i = 0; i < t._shape[0]; i++) {
      os << t(i);
      if (i + 1 != t._shape[0])
        os << ", ";
    }
    os << "]";
    return os;
  }

  if (t._shape.size() == 2) {
    os << "[";
    for (std::size_t i = 0; i < t._shape[0]; i++) {
      os << "[";
      for (std::size_t j = 0; j < t._shape[1]; j++) {
        os << t(i, j);
        if (j + 1 != t._shape[1])
          os << ", ";
      }
      os << "]";
      if (i + 1 != t._shape[0])
        os << ", ";
    }
    os << "]";
    return os;
  }

  os << "<Tensor>";
  return os;
}
