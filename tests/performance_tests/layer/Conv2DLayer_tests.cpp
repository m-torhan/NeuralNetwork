#include <benchmark/benchmark.h>

#include "src/Conv2DLayer.h"
#include "src/Tensor.h"
#include "src/Utils.h"

constexpr uint32_t N = 10;
constexpr uint32_t M = 32;

static void BM_Conv2DLayerForwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M, M, 3 }).applyFunction([](float) { return randNormalDistribution(); });
    Conv2DLayer layer = Conv2DLayer({ M, M, 3 }, 5, 3);

    for (auto _ : state) {
        Tensor c = layer.forwardPropagation(x);
    }
}

static void BM_Conv2DLayerBackwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M, M, 3 }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor dx = Tensor({ N, M, M, 5 }).applyFunction([](float) { return randNormalDistribution(); });
    Conv2DLayer layer = Conv2DLayer({ M, M, 3 }, 5, 3);
    
    layer.initCachedGradient();
    layer.forwardPropagation(x);

    for (auto _ : state) {
        Tensor c = layer.backwardPropagation(dx);
    }
}

BENCHMARK(BM_Conv2DLayerForwardPropagation);
BENCHMARK(BM_Conv2DLayerBackwardPropagation);