#include "tensor.hpp"
#include <functional>
#include <iostream>
#include <stdexcept>

std::vector<std::size_t>
Tensor::compute_stride(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> stride(shape.size());

  if (shape.empty())
    return {};

  stride.back() = 1;

  for (int i = (int)shape.size() - 2; i >= 0; --i) {
    stride[i] = stride[i + 1] * shape[i + 1];
  }

  return stride;
}

std::vector<std::size_t>
Tensor::broadcast_shape(const std::vector<std::size_t> &a,
                        const std::vector<std::size_t> &b) {
  size_t n = std::max(a.size(), b.size());

  std::vector<std::size_t> A(n, 1), B(n, 1);

  for (size_t i = 0; i < a.size(); i++)
    A[n - a.size() + i] = a[i];

  for (size_t i = 0; i < b.size(); i++)
    B[n - b.size() + i] = b[i];

  std::vector<std::size_t> out(n);

  for (size_t i = 0; i < n; i++) {
    if (A[i] == B[i] || A[i] == 1 || B[i] == 1)
      out[i] = std::max(A[i], B[i]);
    else
      throw std::invalid_argument("Broadcast shape mismatch");
  }

  return out;
}

std::size_t Tensor::offset(const std::vector<std::size_t> &idx,
                           const std::vector<std::size_t> &shape,
                           const std::vector<std::size_t> &stride) {
  size_t ndim = shape.size();
  size_t shift = idx.size() - ndim;

  size_t off = 0;

  for (size_t i = 0; i < ndim; i++) {
    size_t dim = shape[i];
    size_t id = idx[shift + i];

    if (dim == 1)
      continue;

    off += id * stride[i];
  }

  return off;
}

Tensor::Tensor(float scalar) {
  _data = {scalar};
  _shape = {};
  _stride = {};
}

Tensor::Tensor(std::vector<float> data) {
  _shape = {data.size()};
  _stride = compute_stride(_shape);
  _data = std::move(data);
}

Tensor::Tensor(std::vector<std::vector<float>> data) {
  size_t rows = data.size();
  size_t cols = data[0].size();

  for (auto &row : data)
    if (row.size() != cols)
      throw std::invalid_argument("Non-rectangular matrix");

  _shape = {rows, cols};
  _stride = compute_stride(_shape);

  _data.reserve(rows * cols);

  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++)
      _data.push_back(data[i][j]);
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }
const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

size_t Tensor::size() const { return _data.size(); }

float &Tensor::operator()(std::size_t i) { return _data[i]; }

const float &Tensor::operator()(std::size_t i) const { return _data[i]; }

float &Tensor::operator()(std::size_t i, std::size_t j) {
  return _data[i * _stride[0] + j];
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const {
  return _data[i * _stride[0] + j];
}

Tensor Tensor::elementwise_op(const Tensor &a, const Tensor &b,
                              std::function<float(float, float)> op) {
  auto out_shape = broadcast_shape(a._shape, b._shape);

  Tensor result;
  result._shape = out_shape;
  result._stride = compute_stride(out_shape);

  size_t total = 1;
  for (auto s : out_shape)
    total *= s;

  result._data.assign(total, 0.0f);

  std::vector<size_t> idx(out_shape.size(), 0);

  for (size_t linear = 0; linear < total; linear++) {

    size_t tmp = linear;

    for (int i = (int)out_shape.size() - 1; i >= 0; i--) {
      idx[i] = tmp % out_shape[i];
      tmp /= out_shape[i];
    }

    size_t a_off = offset(idx, a._shape, a._stride);
    size_t b_off = offset(idx, b._shape, b._stride);

    result._data[linear] = op(a._data[a_off], b._data[b_off]);
  }

  return result;
}

Tensor Tensor::operator+(const Tensor &other) const {
  return elementwise_op(*this, other, [](float x, float y) { return x + y; });
}

Tensor Tensor::operator-(const Tensor &other) const {
  return elementwise_op(*this, other, [](float x, float y) { return x - y; });
}

Tensor Tensor::operator*(const Tensor &other) const {
  return elementwise_op(*this, other, [](float x, float y) { return x * y; });
}

Tensor Tensor::operator/(const Tensor &other) const {
  return elementwise_op(*this, other, [](float x, float y) { return x / y; });
}

Tensor Tensor::matmul(const Tensor &other) const {
  if (_shape.size() != 2 || other._shape.size() != 2)
    throw std::invalid_argument("matmul requires 2D tensors");

  size_t m = _shape[0];
  size_t K = _shape[1];
  size_t n = other._shape[1];

  if (K != other._shape[0])
    throw std::invalid_argument("shape mismatch");

  Tensor result;
  result._shape = {m, n};
  result._stride = compute_stride(result._shape);
  result._data.assign(m * n, 0.0f);

  const float *A = _data.data();
  const float *B = other._data.data();
  float *C = result._data.data();

  for (size_t i = 0; i < m; i++) {
    size_t iK = i * K;
    size_t iN = i * n;

    for (size_t k = 0; k < K; k++) {
      float a = A[iK + k];

      size_t kN = k * n;

      for (size_t j = 0; j < n; j++) {
        C[iN + j] += a * B[kN + j];
      }
    }
  }

  return result;
}

std::ostream &operator<<(std::ostream &os, const Tensor &t) {
  if (t._shape.empty()) {
    os << t._data[0];
    return os;
  }

  if (t._shape.size() == 1) {
    os << "[";
    for (size_t i = 0; i < t._shape[0]; i++) {
      os << t(i);
      if (i + 1 != t._shape[0])
        os << ", ";
    }
    os << "]";
    return os;
  }

  if (t._shape.size() == 2) {
    os << "[";
    for (size_t i = 0; i < t._shape[0]; i++) {
      os << "[";
      for (size_t j = 0; j < t._shape[1]; j++) {
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
