#include "matrix.h"
#include <iostream>

int main() {
  // Create two 2x2 matrices
  Matrix<int> A(2, 2, 0);
  Matrix<int> B(2, 2, 0);

  // Fill matrix A
  A(0, 0) = 1;
  A(0, 1) = 2;
  A(1, 0) = 3;
  A(1, 1) = 4;

  // Fill matrix B
  B(0, 0) = 5;
  B(0, 1) = 6;
  B(1, 0) = 7;
  B(1, 1) = 8;

  // Matrix addition
  Matrix<int> C = A + B;

  std::cout << "A + B:" << std::endl;
  for (unsigned i = 0; i < C.get_rows(); i++) {
    for (unsigned j = 0; j < C.get_cols(); j++) {
      std::cout << C(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // Matrix multiplication
  Matrix<int> D = A * B;

  std::cout << "\nA * B:" << std::endl;
  for (unsigned i = 0; i < D.get_rows(); i++) {
    for (unsigned j = 0; j < D.get_cols(); j++) {
      std::cout << D(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // Transpose
  Matrix<int> T = A.transpose();

  std::cout << "\nTranspose of A:" << std::endl;
  for (unsigned i = 0; i < T.get_rows(); i++) {
    for (unsigned j = 0; j < T.get_cols(); j++) {
      std::cout << T(i, j) << " ";
    }
    std::cout << std::endl;
  }

  // Matrix-vector multiplication
  std::vector<int> v = {1, 2};

  std::vector<int> result = A * v;

  std::cout << "\nA * vector:" << std::endl;
  for (unsigned i = 0; i < result.size(); i++) {
    std::cout << result[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
