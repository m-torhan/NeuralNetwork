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

static void BM_TensorTensorProduct(benchmark::State& state) {
    Tensor a = Tensor({ M, M }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ M, M }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a.tensorProduct(b);
    }
}

static void BM_TensorAddition(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a + b;
    }
}

static void BM_TensorSubtraction(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a - b;
    }
}

static void BM_TensorMultiplication(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a * b;
    }
}

static void BM_TensorDivision(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { float val = randNormalDistribution(); return val != 0 ? val : .25f; });

    for (auto _ : state) {
        Tensor c = a / b;
    }
}

static void BM_TensorCompare(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    Tensor b = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        Tensor c = a < b;
    }
}

static void BM_TensorAdditionScalar(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    float b = randNormalDistribution();

    for (auto _ : state) {
        Tensor c = a + b;
    }
}

static void BM_TensorSubtractionScalar(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    float b = randNormalDistribution();

    for (auto _ : state) {
        Tensor c = a - b;
    }
}

static void BM_TensorMultiplicationScalar(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    float b = randNormalDistribution();

    for (auto _ : state) {
        Tensor c = a * b;
    }
}

static void BM_TensorDivisionScalar(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    float b = randNormalDistribution();
    if (b == 0) {
        b = .25f;
    }

    for (auto _ : state) {
        Tensor c = a / b;
    }
}

static void BM_TensorCompareScalar(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });
    float b = randNormalDistribution();

    for (auto _ : state) {
        Tensor c = a < b;
    }
}

static void BM_TensorSum(benchmark::State& state) {
    Tensor a = Tensor({ N }).applyFunction([](float) { return randNormalDistribution(); });

    for (auto _ : state) {
        float b = a.sum();
    }
}

BENCHMARK(BM_Tensor1D1DDotProduct);
BENCHMARK(BM_Tensor2D1DDotProduct);
BENCHMARK(BM_Tensor2D2DDotProduct);

BENCHMARK(BM_TensorTensorProduct);

BENCHMARK(BM_TensorAddition);
BENCHMARK(BM_TensorSubtraction);
BENCHMARK(BM_TensorMultiplication);
BENCHMARK(BM_TensorDivision);
BENCHMARK(BM_TensorCompare);

BENCHMARK(BM_TensorAdditionScalar);
BENCHMARK(BM_TensorSubtractionScalar);
BENCHMARK(BM_TensorMultiplicationScalar);
BENCHMARK(BM_TensorDivisionScalar);
BENCHMARK(BM_TensorCompareScalar);

BENCHMARK(BM_TensorSum);