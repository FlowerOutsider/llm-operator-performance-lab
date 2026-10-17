#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__         \
                << ": " << cudaGetErrorString(error) << std::endl;         \
      return 1;                                                            \
    }                                                                      \
  } while (0)

int main(int argc, char** argv) {
  int m = 512;
  int n = 512;
  int k = 512;

  if (argc == 4) {
    m = std::atoi(argv[1]);
    n = std::atoi(argv[2]);
    k = std::atoi(argv[3]);
  }

  if (m <= 0 || n <= 0 || k <= 0) {
    std::cerr << "Usage: " << argv[0] << " [M N K]" << std::endl;
    return 1;
  }

  using ElementInputA = cutlass::half_t;
  using ElementInputB = cutlass::half_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    cutlass::half_t,
    cutlass::layout::RowMajor,
    float,                           // FP32 accumulate
    cutlass::arch::OpClassTensorOp,  // Tensor Core
    cutlass::arch::Sm80,             // Ampere Tensor Core 指令族，sm_86 可运行
    ThreadblockShape,
    WarpShape,
    InstructionShape
>;
  std::vector<ElementInputA> host_a(m * k);
  std::vector<ElementInputB> host_b(k * n);
  std::vector<ElementOutput> host_c(m * n, ElementOutput(0));
  std::vector<ElementOutput> host_d(m * n);

  std::mt19937 generator(2026);
  std::uniform_real_distribution<float> distribution(-0.5f, 0.5f);

  for (auto& value : host_a) {
    value = ElementInputA(distribution(generator));
  }
  for (auto& value : host_b) {
    value = ElementInputB(distribution(generator));
  }

  ElementInputA* device_a = nullptr;
  ElementInputB* device_b = nullptr;
  ElementOutput* device_c = nullptr;
  ElementOutput* device_d = nullptr;

  CHECK_CUDA(cudaMalloc(&device_a, host_a.size() * sizeof(ElementInputA)));
  CHECK_CUDA(cudaMalloc(&device_b, host_b.size() * sizeof(ElementInputB)));
  CHECK_CUDA(cudaMalloc(&device_c, host_c.size() * sizeof(ElementOutput)));
  CHECK_CUDA(cudaMalloc(&device_d, host_d.size() * sizeof(ElementOutput)));

  CHECK_CUDA(cudaMemcpy(
      device_a,
      host_a.data(),
      host_a.size() * sizeof(ElementInputA),
      cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      device_b,
      host_b.data(),
      host_b.size() * sizeof(ElementInputB),
      cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(
      device_c,
      host_c.data(),
      host_c.size() * sizeof(ElementOutput),
      cudaMemcpyHostToDevice));

  int lda = k;
  int ldb = n;
  int ldc = n;
  int ldd = n;

  Gemm gemm_op;

  typename Gemm::Arguments arguments(
      {m, n, k},
      {device_a, lda},
      {device_b, ldb},
      {device_c, ldc},
      {device_d, ldd},
      {1.0f, 0.0f});

cutlass::Status status = gemm_op(arguments);
if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM launch failed, status="
              << static_cast<int>(status) << std::endl;

    const cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    }
    return 1;
}

  status = gemm_op(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "CUTLASS GEMM launch failed" << std::endl;
    return 1;
  }

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(
      host_d.data(),
      device_d,
      host_d.size() * sizeof(ElementOutput),
      cudaMemcpyDeviceToHost));

  float max_abs_error = 0.0f;
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float reference = 0.0f;
      for (int inner = 0; inner < k; ++inner) {
        reference += static_cast<float>(host_a[row * k + inner]) *
                     static_cast<float>(host_b[inner * n + col]);
      }

      float actual = static_cast<float>(host_d[row * n + col]);
      max_abs_error = std::max(
          max_abs_error,
          std::abs(reference - actual));
    }
  }

  constexpr int warmup_iterations = 20;
  constexpr int benchmark_iterations = 100;

  for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
    status = gemm_op(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS GEMM warmup failed" << std::endl;
      return 1;
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start_event;
  cudaEvent_t end_event;
  CHECK_CUDA(cudaEventCreate(&start_event));
  CHECK_CUDA(cudaEventCreate(&end_event));

  CHECK_CUDA(cudaEventRecord(start_event));
  for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
    status = gemm_op(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "CUTLASS GEMM benchmark failed" << std::endl;
      return 1;
    }
  }
  CHECK_CUDA(cudaEventRecord(end_event));
  CHECK_CUDA(cudaEventSynchronize(end_event));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));

  float average_ms = elapsed_ms / benchmark_iterations;
  double tflops =
      (2.0 * static_cast<double>(m) * n * k) /
      (static_cast<double>(average_ms) * 1e-3) /
      1e12;

  std::cout << "CUTLASS FP16 Tensor Core GEMM" << std::endl;
  std::cout << "M=" << m << " N=" << n << " K=" << k << std::endl;
  std::cout << "max_abs_error=" << max_abs_error << std::endl;
  std::cout << "average_ms=" << average_ms << std::endl;
  std::cout << "tflops=" << tflops << std::endl;

  CHECK_CUDA(cudaEventDestroy(start_event));
  CHECK_CUDA(cudaEventDestroy(end_event));
  CHECK_CUDA(cudaFree(device_a));
  CHECK_CUDA(cudaFree(device_b));
  CHECK_CUDA(cudaFree(device_c));
  CHECK_CUDA(cudaFree(device_d));

  return max_abs_error < 0.5f ? 0 : 1;
}