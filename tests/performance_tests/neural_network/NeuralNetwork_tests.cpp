#include <benchmark/benchmark.h>

#include "src/Tensor.h"
#include "src/DenseLayer.h"
#include "src/ActivationLayer.h"
#include "src/NeuralNetwork.h"
#include "src/Utils.h"

constexpr uint32_t N = 1000;
constexpr uint32_t M = 100;

static void BM_NeuralNetworkPredict(benchmark::State& state) {
	uint32_t i = 0;

	Tensor x_test = Tensor({ M, 2 });
	Tensor y_test = Tensor({ M, 2 });

	srand(time(NULL));

	for (i = 0; i < x_test.getShape()[0]; ++i) {
		float x = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
		float y = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;

		x_test.setValue(x, { i, 0 });
		x_test.setValue(y, { i, 1 });

		float u = (x * x + y * y < (2.0f / 3.1415f)) ? 1.0f : 0.0f;

		y_test.setValue(u, { i, 0 });
		y_test.setValue(1 - u, { i, 1 });
	}

	auto layer_1 = DenseLayer({ 2 }, 16);
	auto layer_2 = ActivationLayer(layer_1, ActivationFun::LeakyReLU);
	auto layer_3 = DenseLayer(layer_2, 16);
	auto layer_4 = ActivationLayer(layer_3, ActivationFun::LeakyReLU);
	auto layer_5 = DenseLayer(layer_4, 16);
	auto layer_6 = ActivationLayer(layer_5, ActivationFun::LeakyReLU);
	auto layer_7 = DenseLayer(layer_6, 2);
	auto layer_8  = ActivationLayer(layer_7, ActivationFun::Sigmoid);

	auto nn = NeuralNetwork(layer_1, layer_8, CostFun::BinaryCrossentropy);

    for (auto _ : state) {
	    Tensor y_hat = nn.predict(x_test);
    }
}

static void BM_NeuralNetworkFit(benchmark::State& state) {
	uint32_t i = 0;

	Tensor x_train = Tensor({ N, 2 });
	Tensor y_train = Tensor({ N, 2 });
	Tensor x_test = Tensor({ M, 2 });
	Tensor y_test = Tensor({ M, 2 });

	srand(time(NULL));

	for (i = 0; i < x_train.getShape()[0]; ++i) {
		float x = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
		float y = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;

		x_train.setValue(x, { i, 0 });
		x_train.setValue(y, { i, 1 });

		float u = (x * x + y * y < (2.0f / 3.1415f)) ? 1.0f : 0.0f;

		y_train.setValue(u, { i, 0 });
		y_train.setValue(1 - u, { i, 1 });
	}

	for (i = 0; i < x_test.getShape()[0]; ++i) {
		float x = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;
		float y = (static_cast<float>(rand()) / RAND_MAX) * 2 - 1;

		x_test.setValue(x, { i, 0 });
		x_test.setValue(y, { i, 1 });

		float u = (x * x + y * y < (2.0f / 3.1415f)) ? 1.0f : 0.0f;

		y_test.setValue(u, { i, 0 });
		y_test.setValue(1 - u, { i, 1 });
	}

	auto layer_1 = DenseLayer({ 2 }, 16);
	auto layer_2 = ActivationLayer(layer_1, ActivationFun::LeakyReLU);
	auto layer_3 = DenseLayer(layer_2, 16);
	auto layer_4 = ActivationLayer(layer_3, ActivationFun::LeakyReLU);
	auto layer_5 = DenseLayer(layer_4, 16);
	auto layer_6 = ActivationLayer(layer_5, ActivationFun::LeakyReLU);
	auto layer_7 = DenseLayer(layer_6, 2);
	auto layer_8  = ActivationLayer(layer_7, ActivationFun::Sigmoid);
    
	auto nn = NeuralNetwork(layer_1, layer_8, CostFun::BinaryCrossentropy);

    for (auto _ : state) {
	    nn.fit(x_train, y_train, x_test, y_test, 500, 20, 0.05f, 0);
    }
}

BENCHMARK(BM_NeuralNetworkPredict);
BENCHMARK(BM_NeuralNetworkFit);