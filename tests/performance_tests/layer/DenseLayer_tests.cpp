#include <benchmark/benchmark.h>

#include "src/DenseLayer.h"
#include "src/Tensor.h"
#include "src/Utils.h"

constexpr uint32_t N = 100;
constexpr uint32_t M = 100;

static void BM_DenseLayerForwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    DenseLayer layer = DenseLayer({ M }, M);

    for (auto _ : state) {
        Tensor c = layer.forwardPropagation(x);
    }
}

static void BM_DenseLayerBackwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor dx = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    DenseLayer layer = DenseLayer({ M }, M);
    
    layer.initCachedGradient();
    layer.forwardPropagation(x);

    for (auto _ : state) {
        Tensor c = layer.backwardPropagation(dx);
    }
}

BENCHMARK(BM_DenseLayerForwardPropagation);
BENCHMARK(BM_DenseLayerBackwardPropagation);