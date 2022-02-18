#pragma once

#include "Layer.h"

typedef struct FitHistory {
	float* train_cost;
	float* test_cost;
};

class NeuralNetwork {
public:
	NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&));

	Tensor* predict(Tensor* input);
	FitHistory* fit(Tensor* train_x, Tensor* train_y, Tensor* test_x, Tensor* test_y, uint32_t batch_size, uint32_t epochs, float learning_step);

private:
	Layer* _input_layer;
	Layer* _output_layer;
	float(*_cost_function)(const Tensor&, const Tensor&);
};