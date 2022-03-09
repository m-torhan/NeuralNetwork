#include "NeuralNetwork.h"
#include "ActivationLayer.h"
#include "DenseLayer.h"

#include <cstdint>
#include <stdio.h>
#include <windows.h>
#include <ctime>
#include <cmath>

constexpr int N = 10000;
constexpr int M = 1000;

int main() {
	uint32_t i = 0;
	uint32_t j = 0;

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

	printf("Network init\n");

	auto layer_1 = DenseLayer({ 2 }, 16);
	auto layer_2 = ActivationLayer(layer_1, ActivationFun::LeakyReLU);
	auto layer_3 = DenseLayer(layer_2, 16);
	auto layer_4 = ActivationLayer(layer_3, ActivationFun::LeakyReLU);
	auto layer_5 = DenseLayer(layer_4, 16);
	auto layer_6 = ActivationLayer(layer_5, ActivationFun::LeakyReLU);
	auto layer_7 = DenseLayer(layer_6, 16);
	auto layer_8 = ActivationLayer(layer_7, ActivationFun::LeakyReLU);
	auto layer_9 = DenseLayer(layer_8, 16);
	auto layer_10 = ActivationLayer(layer_9, ActivationFun::LeakyReLU);
	auto layer_11 = DenseLayer(layer_10, 2);
	auto layer_12  = ActivationLayer(layer_11, ActivationFun::Sigmoid);
	auto nn = NeuralNetwork(layer_1, layer_12, CostFun::BinaryCrossentropy);

	Tensor y_hat = nn.predict(x_test);

	float cost_1 = nn.getCostFun()(y_hat, y_test);

	float correct_1 = 0;
	for (i = 0; i < y_test.getShape()[0]; ++i) {
		if ((y_test.getValue({ i, 0 }) > y_test.getValue({ i, 1 }) && y_hat.getValue({ i, 0 }) > y_hat.getValue({ i, 1 })) ||
			(y_test.getValue({ i, 0 }) < y_test.getValue({ i, 1 }) && y_hat.getValue({ i, 0 }) < y_hat.getValue({ i, 1 }))) {
			correct_1 += 1.0f;
		}
	}

	nn.fit(x_train, y_train, x_test, y_test, 500, 20, 0.05f);

	y_hat = nn.predict(x_test);
	
	float cost_2 = nn.getCostFun()(y_hat, y_test);

	float correct_2 = 0;
	for (i = 0; i < y_test.getShape()[0]; ++i) {
		if ((y_test.getValue({ i, 0 }) > y_test.getValue({ i, 1 }) && y_hat.getValue({ i, 0 }) > y_hat.getValue({ i, 1 })) ||
			(y_test.getValue({ i, 0 }) < y_test.getValue({ i, 1 }) && y_hat.getValue({ i, 0 }) < y_hat.getValue({ i, 1 }))) {
			correct_2 += 1.0f;
		}
	}
	printf("cost: %f \tscore: %f\n", cost_1, correct_1 / y_test.getShape()[0]);
	printf("cost: %f \tscore: %f\n", cost_2, correct_2 / y_test.getShape()[0]);
}