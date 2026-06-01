#include "tensor.hpp"
#include <chrono>
#include <iostream>
#include <vector>

void bench_matmul(int N, int ITER) {
  std::cout << "Matrix size: " << N << "x" << N << "\n";
  std::cout << "Iterations: " << ITER << "\n";

  Tensor A(std::vector<std::vector<float>>(N, std::vector<float>(N, 1.0f)));

  Tensor B(std::vector<std::vector<float>>(N, std::vector<float>(N, 2.0f)));

  // warmup (important to avoid cold start noise)
  volatile Tensor tmp = A.matmul(B);

  auto start = std::chrono::high_resolution_clock::now();

  Tensor C;

  for (int i = 0; i < ITER; i++) {
    C = A.matmul(B);
  }

  auto end = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double, std::milli> ms = end - start;

  double avg = ms.count() / ITER;

  std::cout << "Total time: " << ms.count() << " ms\n";
  std::cout << "Avg per matmul: " << avg << " ms\n\n";
}

void test_broadcasting() {
  std::cout << "=== Broadcasting tests ===\n";

  Tensor A(std::vector<std::vector<float>>{{1, 2, 3}, {4, 5, 6}});

  Tensor v(std::vector<float>{10, 20, 30});

  Tensor scalar(5.0f);

  std::cout << "A:\n" << A << "\n\n";
  std::cout << "v:\n" << v << "\n\n";

  auto Av = A + v; // (2,3) + (3,) broadcast
  std::cout << "A + v = " << Av << "\n\n";

  auto As = A + scalar; // scalar broadcast
  std::cout << "A + scalar = " << As << "\n\n";
}

void correctness_checks() {
  std::cout << "=== Correctness checks ===\n";

  Tensor A(std::vector<float>{1, 2, 3});
  Tensor B(std::vector<float>{4, 5, 6});

  // auto dot = A.matmul(B); // if you support 1D matmul or adapt later

  std::cout << "A: " << A << "\n";
  std::cout << "B: " << B << "\n";
  // std::cout << "dot: " << dot << "\n\n";

  Tensor M1(std::vector<std::vector<float>>{{1, 2}, {3, 4}});

  Tensor M2(std::vector<std::vector<float>>{{5, 6}, {7, 8}});

  std::cout << "M1:\n" << M1 << "\n";
  std::cout << "M2:\n" << M2 << "\n";

  auto M3 = M1 + M2;
  std::cout << "M1 + M2 = " << M3 << "\n";

  auto M4 = M1.matmul(M2);
  std::cout << "M1 @ M2 = " << M4 << "\n\n";
}

int main() {
  correctness_checks();
  test_broadcasting();
  bench_matmul(300, 20); // adjust size for your machine
  return 0;
}
