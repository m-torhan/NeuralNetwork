#include "NeuralNetwork.h"

#include <cstdlib>

NeuralNetwork::NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&)) {
	_input_layer = &input_layer;
	_output_layer = &output_layer;
	_cost_function = cost_function;
}

Tensor* NeuralNetwork::predict(Tensor* input) {
	Layer* layer;
	Tensor* output;

	layer = _input_layer;
	output = layer->forwardPropagation(*input);

	while (layer != _output_layer) {
		output = layer->forwardPropagation(*output);
		layer = layer->getNextLayer();
	}

	return output;
}

FitHistory* NeuralNetwork::fit(Tensor* train_x, Tensor* train_y, Tensor* test_x, Tensor* test_y, uint32_t batch_size, uint32_t epochs, float learning_step) {
	FitHistory* result;
	uint32_t epoch = 0;

	result = (FitHistory*)malloc(sizeof(FitHistory));
	if (!result) {
		// exception
	}
	result->test_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result->test_cost) {
		// exception
	}
	result->train_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result->train_cost) {
		// exception
	}

	for (epoch = 0; epoch < epochs; ++epoch) {

	}

	return result;
}