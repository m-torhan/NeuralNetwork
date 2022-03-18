#include <benchmark/benchmark.h>

#include "src/ActivationLayer.h"
#include "src/Tensor.h"
#include "src/Utils.h"

constexpr uint32_t N = 100;
constexpr uint32_t M = 100;

static void BM_ActivationLayerSigmoidForwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::Sigmoid);

    for (auto _ : state) {
        Tensor c = layer.forwardPropagation(x);
    }
}

static void BM_ActivationLayerSigmoidBackwardPropagation(benchmark::State& state) {
    Tensor dx = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::Sigmoid);

    for (auto _ : state) {
        Tensor c = layer.backwardPropagation(dx);
    }
}

static void BM_ActivationLayerReLUForwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::ReLU);

    for (auto _ : state) {
        Tensor c = layer.forwardPropagation(x);
    }
}

static void BM_ActivationLayerReLUBackwardPropagation(benchmark::State& state) {
    Tensor dx = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::ReLU);

    for (auto _ : state) {
        Tensor c = layer.backwardPropagation(dx);
    }
}

static void BM_ActivationLayerLeakyReLUForwardPropagation(benchmark::State& state) {
    Tensor x = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::LeakyReLU);

    for (auto _ : state) {
        Tensor c = layer.forwardPropagation(x);
    }
}

static void BM_ActivationLayerLeakyReLUBackwardPropagation(benchmark::State& state) {
    Tensor dx = Tensor({ N, M }).applyFunction([](float) { return randNormalDistribution(); });
    ActivationLayer layer = ActivationLayer({ M }, ActivationFun::LeakyReLU);

    for (auto _ : state) {
        Tensor c = layer.backwardPropagation(dx);
    }
}

BENCHMARK(BM_ActivationLayerSigmoidForwardPropagation);
BENCHMARK(BM_ActivationLayerSigmoidBackwardPropagation);
BENCHMARK(BM_ActivationLayerReLUForwardPropagation);
BENCHMARK(BM_ActivationLayerReLUBackwardPropagation);
BENCHMARK(BM_ActivationLayerLeakyReLUForwardPropagation);
BENCHMARK(BM_ActivationLayerLeakyReLUBackwardPropagation);