#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

enum class DType {
  Float32,
  Float64,
  Int32,
  Int64,
};

class Tensor {
private:
  std::vector<float> _data;
  std::vector<int64_t> _shape;
  std::vector<int64_t> _strides;
  DType _dtype;
  int64_t _data_size;

  std::vector<int64_t> compute_strides(const std::vector<int64_t> &_shape);

public:
  // Constructor
  Tensor();
  Tensor(const std::vector<int64_t> &_shape, DType dtype = DType::Float32);
  Tensor(std::vector<float> &&_data, const std::vector<int64_t> &_shape,
         DType dtype = DType::Float32);

  // Factory methods
  static Tensor zeros(const std::vector<int64_t> &_shape);
  static Tensor ones(const std::vector<int64_t> &_shape);
  static Tensor from_vector(const std::vector<float> &_data,
                            const std::vector<int64_t> &_shape);
  static Tensor arange(float start, float end, float step = 1.0);

  // Accessors
  int64_t numel() const { return _data_size; }
  const std::vector<int64_t> &get_shape() const { return _shape; }
  const std::vector<int64_t> &get_strides() const { return _strides; }
  int dim() const { return _shape.size(); }
  DType get_dtype() const { return _dtype; }
  float *get_data() { return _data.data(); }
  const float *get_data() const { return _data.data(); }

  // Indexing
  float &operator()(const std::vector<int64_t> &indices);
  float operator()(const std::vector<int64_t> &indices) const;

  // Shape manipulation
  Tensor reshape(const std::vector<int64_t> &new_shape);
  Tensor transpose() const;
  Tensor unsqueeze(int dim) const;
  Tensor squeeze(int dim) const;

  // Element-wise operations
  Tensor operator+(const Tensor &other) const;
  Tensor operator-(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;
  Tensor operator/(const Tensor &other) const;

  Tensor operator+(float scalar) const;
  Tensor operator*(float scalar) const;

  // In-place operations
  void add_(const Tensor &other);
  void mul_(float scalar);

  // Reductions
  float sum() const;
  float mean() const;
  float max() const;
  float min() const;

  Tensor sum(int axis) const;
  Tensor mean(int axis) const;

  void print() const;

  int64_t compute_offset(const std::vector<int64_t> &indices) const;
};
