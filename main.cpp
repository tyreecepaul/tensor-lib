#include "tensor.hpp"
#include <chrono>
#include <iostream>

int main() {
  int N{1024};
  int ITER{50};

  Tensor A(std::vector<std::vector<float>>(N, std::vector<float>(N, 1.0f)));

  Tensor B(std::vector<std::vector<float>>(N, std::vector<float>(N, 2.0f)));

  auto start = std::chrono::high_resolution_clock::now();

  Tensor C = A * B;

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> ms = end - start;

  std::cout << "Time: " << ms.count() << " ms\n";
}
