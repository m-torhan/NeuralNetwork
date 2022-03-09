#include <benchmark/benchmark.h>

#include "src/Tensor.h"
#include "src/Utils.h"

static void BM_TensorDotProduct(benchmark::State& state) {
  Tensor a = Tensor({1, 1000}).applyFunction([](float) {return randNormalDistribution(); });
  Tensor b = Tensor({1, 1000}).applyFunction([](float) {return randNormalDistribution(); });
  for (auto _ : state) {
    Tensor c = a * b;
  }
}

BENCHMARK(BM_TensorDotProduct);