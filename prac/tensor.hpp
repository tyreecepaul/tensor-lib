#pragma once

#include <cstddef>
#include <functional>
#include <iostream>
#include <vector>

class Tensor {
private:
  std::vector<float> _data;
  std::vector<std::size_t> _shape;
  std::vector<std::size_t> _stride;

private:
  static std::vector<std::size_t>
  compute_stride(const std::vector<std::size_t> &shape);

  static std::vector<std::size_t>
  broadcast_shape(const std::vector<std::size_t> &a,
                  const std::vector<std::size_t> &b);

  static std::size_t offset(const std::vector<std::size_t> &idx,
                            const std::vector<std::size_t> &shape,
                            const std::vector<std::size_t> &stride);

  static Tensor elementwise_op(const Tensor &a, const Tensor &b,
                               std::function<float(float, float)> op);

public:
  // --- constructors ---
  Tensor() = default;

  Tensor(float scalar);

  Tensor(std::vector<float> data);

  Tensor(std::vector<std::vector<float>> data);

  // --- shape info ---
  const std::vector<std::size_t> &shape() const;
  const std::vector<std::size_t> &stride() const;

  std::size_t size() const;

  // --- indexing ---
  float &operator()(std::size_t i);
  float &operator()(std::size_t i, std::size_t j);

  const float &operator()(std::size_t i) const;
  const float &operator()(std::size_t i, std::size_t j) const;

  // --- elementwise ops (broadcasting-enabled) ---
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;

  // --- matrix multiplication (separate kernel) ---
  Tensor matmul(const Tensor &other) const;

  // --- output ---
  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};
