#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

class Tensor {
private:
  std::vector<float> _data;
  std::vector<std::size_t> _shape;
  std::vector<std::size_t> _stride;

public:
  // constructors
  Tensor() = default;
  Tensor(float scalar);
  Tensor(std::vector<float> data);
  Tensor(std::vector<std::vector<float>> data);

  // access
  float &operator()(std::size_t i);
  float &operator()(std::size_t i, std::size_t j);

  const float &operator()(std::size_t i) const;
  const float &operator()(std::size_t i, std::size_t j) const;

  // shape
  const std::vector<std::size_t> &shape() const;

  // ops (VALUE SEMANTICS)
  Tensor operator+(const Tensor &other) const;
  Tensor operator*(const Tensor &other) const;

  // print
  friend std::ostream &operator<<(std::ostream &os, const Tensor &t);
};
