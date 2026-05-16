#include <iostream>
#include <vector>

template <typename T> class Matrix {
private:
  unsigned rows;
  unsigned cols;
  std::vector<T> data;
  size_t index(unsigned row, unsigned col) const { return row * cols + col; }

public:
  Matrix(unsigned r, unsigned c, const T &init = T{})
      : rows(r), cols(c), data(r * c, init) {}
  T &operator()(unsigned row, unsigned col) { return data[index(row, col)]; }
  const T &operator()(unsigned row, unsigned col) const {
    return data[index(row, col)];
  }
};

int main() {
  Matrix<int> m(3, 4, 0);
  m(1, 2) = 99;
  std::cout << m(1, 2) << '\n';
};
