#include "tensor.hpp"
#include <memory>
#include <ostream>
#include <stdexcept>
#include <vector>

Tensor::Tensor(float data) : _data{data}, _shape{}, _stride{} {};

Tensor::Tensor(std::vector<float> data)
    : _data(data), _shape{data.size()}, _stride{1} {};

Tensor::Tensor(std::vector<std::vector<float>> data)
    : _shape{data.size(), data[0].size(), 1} {
  // check if dimensions match
  std::size_t n_expected_columns = data[0].size();
  for (std::size_t i = 0; i < data.size(); i++) {
    if (data[i].size() != n_expected_columns) {
      throw std::invalid_argument("Dimensions are inconsistent.");
    }
  }

  // store in row major format
  for (std::size_t i = 0; i < data.size(); i++) {
    for (std::size_t j = 0; j < data.size(); j++) {
      _data.push_back(data[i][j]);
    }
  }
}

const float &Tensor::item() const {
  if (_data.size() == 1) {
    return _data[0];
  } else {
    throw std::runtime_error(
        "item() can only be called on tensors with a single element");
  }
}

const float &Tensor::operator()(std::size_t i) const {
  if (_shape.size() == 0) {
    throw std::invalid_argument(
        "Can't index into a scalar, use item() instead");
  }
  if (_shape.size() == 1) {
    if (i >= _shape[0]) {
      throw std::invalid_argument("Index " + std::to_string(i) +
                                  " is out of bounds for array of size " +
                                  std::to_string(_shape[0]) + " rows");
    }
    return _data[i];
  }
  throw std::invalid_argument(
      "This is a 1D tensor. Use two indices for 2D tensors.");
}

const float &Tensor::operator()(std::size_t i, std::size_t j) const {
  if (_shape.size() == 2) {
    if (i >= _shape[0]) {
      throw std::invalid_argument("Row index " + std::to_string(i) +
                                  "is out of bounds for array of size " +
                                  std::to_string(_shape[0]) + " rows");
    }
    if (j >= _shape[1]) {
      throw std::invalid_argument("Row index " + std::to_string(i) +
                                  "is out of bounds for array of size " +
                                  std::to_string(_shape[0]) + " rows");
    }

    return _data[i * _stride[0] * j * _stride[1]];
  }
  throw std::invalid_argument("Can only double index into 2D tensors");
}

const std::vector<std::size_t> &Tensor::shape() const { return _shape; }

const std::vector<std::size_t> &Tensor::stride() const { return _stride; }

std::ostream &operator<<(std::ostream &os, const Tensor &obj) {
  std::string string_repr;

  if (obj.shape().empty()) {
    os << obj.item();
    return os;
  }

  // 1D tensor
  if (obj.shape().size() == 1) {
    string_repr += "[";
    for (std::size_t i = 0; i < obj.shape()[0]; i++) {
      string_repr += std::to_string(obj(i));
      if (i != obj.shape()[0] - 1) {
        string_repr += ", ";
      }
    }
    string_repr += "]";
    os << string_repr;
    return os;
  }

  // 2D tensor
  if (obj.shape().size() == 2) {
    string_repr += "[";
    for (std::size_t i = 0; i < obj.shape()[0]; i++) {
      string_repr += "[";
      for (std::size_t j = 0; j < obj.shape()[1]; j++) {
        string_repr += std::to_string(obj(i, j));
        if (j != obj.shape()[1] - 1) {
          string_repr += ", ";
        }
      }
      string_repr += "]";
      if (i != obj.shape()[0] - 1) {
        string_repr += ", ";
      }
    }
    string_repr += "]";
    os << string_repr;
    return os;
  }

  // fallback (higher dims not implemented yet)
  os << "<Tensor shape=";
  for (auto s : obj.shape())
    os << s << " ";
  os << ">";
  return os;
}

std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
  // scalar + scalar
  if (_shape.size() == 0 && other->shape().size() == 0) {
    float res = item() + other->item();
    return std::make_shared<Tensor>(res);
  }

  // scalar + 1D
  if (_shape.size() == 0 && other->shape().size() == 1) {
    std::vector<float> res;
    for (std::size_t i = 0; i < other->shape()[0]; i++) {
      res.push_back(item() + ((*other)(i)));
    }
    return std::make_shared<Tensor>(res);
  }

  // scalar + 2D
  if (_shape.size() == 0 && other->shape().size() == 2) {
    std::vector<std::vector<float>> res;
    for (std::size_t i = 0; i < other->shape()[0]; i++) {
      std::vector<float> res_i;
      for (std::size_t j = 0; j < other->shape()[1]; j++) {
        res_i.push_back(item() + (*other)(i, j));
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }

  // 1D + scalar
  if (_shape.size() == 1 && other->shape().size() == 0) {
    std::vector<float> res;
    for (std::size_t i = 0; i < other->shape()[1]; i++) {
      res.push_back(operator()(i) + other->item());
    }
    return std::make_shared<Tensor>(res);
  }

  // 2D + scalar
  if (_shape.size() == 2 && other->shape().size() == 1) {
    std::vector<std::vector<float>> res;
    for (std::size_t i = 0; i < other->shape()[0]; i++) {
      std::vector<float> res_i;
      for (std::size_t j = 0; j < other->shape()[1]; j++) {
        res_i.push_back(operator()(i, j) + other->item());
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }

  // 1D + 1D
  if (_shape[0] != other->shape()[0]) {
    throw std::invalid_argument("First dimensions not equal");
  }

  if (_shape.size() == 1) {
    std::vector<float> res;
    for (std::size_t i = 0; i < shape()[0]; i++) {
      res.push_back(operator()(i) + (*other)(i));
    }
    return std::make_shared<Tensor>(res);
  }

  // 2D + 2D
  else {
    if (shape()[1] != other->shape()[1]) {
      throw std::invalid_argument("Second dimensions are not equal");
    }
    std::vector<std::vector<float>> res;
    for (std::size_t i = 0; i < shape()[0]; i++) {
      std::vector<float> res_i;
      for (std::size_t j = 0; j < shape()[1]; j++) {
        res_i.push_back(operator()(i, j) + (*other)(i, j));
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
  if (_shape.size() == 0 || other->shape().size() == 0) {
    throw std::invalid_argument(
        "Both arguments need to be at least 1D for matmul");
  }
  if (_shape[_shape.size() - 1] != other->shape()[0]) {
    throw std::invalid_argument("Last dimension of first tensor does not have "
                                "same size as first dimension of second");
  }

  // 1D x 1D -> float
  if (_shape.size() == 1 && other->shape().size() == 1) {
    float res = 0;
    for (std::size_t i = 0; i < _shape[0]; i++) {
      res += operator()(i) * (*other)(i);
    }
    return std::make_shared<Tensor>(res);
  }

  // 2D x 1D -> 1D
  else if (_shape.size() == 2 && other->shape().size() == 1) {
    std::vector<float> res;
    for (std::size_t i = 0; i < _shape[0]; i++) {
      float res_i = 0.0f;
      for (std::size_t j = 0; j < _shape[1]; j++) {
        res_i += operator()(i, j) * (*other)(j);
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }

  // 1D x 2D -> 1D
  else if (_shape.size() == 1 && other->shape().size() == 2) {
    std::vector<float> res;
    for (std::size_t i = 0; i < other->shape()[1]; i++) {
      float res_i = 0.0f;
      for (std::size_t j = 0; j < other->shape()[0]; j++) {
        res_i += operator()(j) * (*other)(i, j);
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }

  // 2D x 2D -> 2D
  else {
    if (other->shape().size() < 2) {
      throw std::invalid_argument("Expected second tensor to have at least 2 "
                                  "dimensions for this operation");
    }
    std::vector<std::vector<float>> res;
    for (std::size_t i = 0; i < shape()[0]; i++) {
      std::vector<float> res_i;
      for (std::size_t j = 0; j < other->shape()[1]; j++) {
        float res_i_j = 0.0f;
        for (std::size_t k = 0; k < shape()[1]; k++) {
          res_i_j += operator()(i, k) * (*other)(k, j);
        }
        res_i.push_back(res_i_j);
      }
      res.push_back(res_i);
    }
    return std::make_shared<Tensor>(res);
  }
}
