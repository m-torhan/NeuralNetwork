#include "NeuralNetwork.h"
#include "ActivationLayer.h"
#include "DenseLayer.h"

#include <cstdint>
#include <stdio.h>
#include <windows.h>
#include <ctime>
#include <cmath>

constexpr int N = 10000;

int main() {
	uint32_t i = 0;
	Tensor* x_train;
	Tensor* y_train;
	Tensor* x_test;
	Tensor* y_test;
	Tensor* y_hat;
	Tensor* (*activation_fun)(const Tensor & x) = [](const Tensor& x) { return x * x; };
	Tensor* (*activation_fun_d)(const Tensor & x, const Tensor & dx) = [](const Tensor& x, const Tensor& dx) { return x * (*(dx * 2.0f)); };
	ActivationLayer* layer_1;
	DenseLayer* layer_2;
	ActivationLayer* layer_3;
	DenseLayer* layer_4;
	ActivationLayer* layer_5;
	DenseLayer* layer_6;
	ActivationLayer* layer_7;
	DenseLayer* layer_8;
	ActivationLayer* layer_9;
	NeuralNetwork* nn;
	float x;
	float y;
	float u;
	float cost_1;
	float cost_2;

	x_train = new Tensor(2, N, 2);
	y_train = new Tensor(2, N, 2);
	x_test = new Tensor(2, 64, 2);
	y_test = new Tensor(2, 64, 2);

	srand(time(NULL));

	for (i = 0; i < x_train->getShape()[0]; ++i) {
		x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
		y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

		x_train->setValue(x, 2, i, 0);
		x_train->setValue(y, 2, i, 1);

		u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

		y_train->setValue(u, 2, i, 0);
		y_train->setValue(1.0f - u, 2, i, 1);
	}
	for (i = 0; i < x_test->getShape()[0]; ++i) {
		x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
		y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

		x_test->setValue(x, 2, i, 0);
		x_test->setValue(y, 2, i, 1);

		u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

		y_test->setValue(u, 2, i, 0);
		y_test->setValue(1.0f - u, 2, i, 1);
	}

	layer_1 = new ActivationLayer(x_train->getDim() - 1, &x_train->getShape()[1], ActivationFun::ReLU);
	layer_2 = new DenseLayer(*layer_1, 128);
	layer_3 = new ActivationLayer(*layer_2, ActivationFun::ReLU);
	layer_4 = new DenseLayer(*layer_3, 128);
	layer_5 = new ActivationLayer(*layer_4, ActivationFun::ReLU);
	layer_6 = new DenseLayer(*layer_5, 128);
	layer_7 = new ActivationLayer(*layer_6, ActivationFun::ReLU);
	layer_8 = new DenseLayer(*layer_7, 2);
	layer_9 = new ActivationLayer(*layer_8, ActivationFun::Sigmoid);

	nn = new NeuralNetwork(*layer_1, *layer_9, CostFun::BinaryCrossentropy);

	y_hat = nn->predict(x_test);

	cost_1 = nn->getCostFun()(*y_hat, *y_test);

	float correct_1 = 0;
	for (i = 0; i < y_test->getShape()[0]; ++i) {
		if ((y_test->getValue(2, i, 0) > y_test->getValue(2, i, 1) && y_hat->getValue(2, i, 0) > y_hat->getValue(2, i, 1)) ||
			(y_test->getValue(2, i, 0) < y_test->getValue(2, i, 1) && y_hat->getValue(2, i, 0) < y_hat->getValue(2, i, 1))) {
			correct_1 += 1.0f;
		}
	}

	nn->fit(x_train, y_train, x_test, y_test, 64, 60, 1.0f);

	y_hat = nn->predict(x_test);

	cost_2 = nn->getCostFun()(*y_hat, *y_test);

	float correct_2 = 0;
	for (i = 0; i < y_test->getShape()[0]; ++i) {
		if ((y_test->getValue(2, i, 0) > y_test->getValue(2, i, 1) && y_hat->getValue(2, i, 0) > y_hat->getValue(2, i, 1)) ||
			(y_test->getValue(2, i, 0) < y_test->getValue(2, i, 1) && y_hat->getValue(2, i, 0) < y_hat->getValue(2, i, 1))) {
			correct_2 += 1.0f;
		}
	}
	printf("cost: %f \tscore: %f\n", cost_1, correct_1 / y_test->getShape()[0]);
	printf("cost: %f \tscore: %f\n", cost_2, correct_2 / y_test->getShape()[0]);
	
	delete x_train;
	delete y_train;
	delete y_hat;
	delete layer_1;
	delete layer_2;
	delete layer_3;
	delete layer_4;
	delete layer_5;
	delete layer_6;
	delete layer_7;
	delete nn;
}