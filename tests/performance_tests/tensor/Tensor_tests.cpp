#include <benchmark/benchmark.h>

#include "src/Tensor.h"
#include "src/Utils.h"

constexpr uint32_t N = 10000;
constexpr uint32_t M = 100;

static void BM_Tensor1D1DDotProduct(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    
    for (auto _ : state) {
        Tensor c = a.dotProduct(b);
    }
}

static void BM_Tensor2D1DDotProduct(benchmark::State& state) {
    Tensor a = Tensor({ M, N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a.dotProduct(b);
    }
}

static void BM_Tensor2D2DDotProduct(benchmark::State& state) {
    Tensor a = Tensor({ M, M }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ M, M }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a.dotProduct(b);
    }
}

BENCHMARK(BM_Tensor1D1DDotProduct);
BENCHMARK(BM_Tensor2D1DDotProduct);
BENCHMARK(BM_Tensor2D2DDotProduct);