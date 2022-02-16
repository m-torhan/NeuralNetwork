#include "DenseLayer.h"

#include <cstdlib>
#include <cstring>

DenseLayer::DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count) {
	InitInput(input_dim, input_shape);
}

DenseLayer::DenseLayer(const Layer& prev_layer, uint32_t neurons_count) {
	InitInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
}

Tensor* DenseLayer::forwardPropagation(const Tensor& x) {
	Tensor* x_next = x.flatten();
	x_next = *_weights->dotProduct(*x_next) + *_biases;
	_cached_output = new Tensor(*x_next);
	return x_next;
}

Tensor* DenseLayer::backwardPropagation(const Tensor& dx) {
	uint32_t n;
	Tensor* dx_prev = new Tensor(dx);
	Tensor* weights_d;
	Tensor* biases_d;

	n = _cached_output->getShape()[0];

	weights_d = *dx.dotProduct(*_cached_output->transpose()) / n;
	biases_d = *dx.sum(1) / n;

	dx_prev = _weights->dotProduct(dx);

	// _weights -= weights_d * learning_step;
	// _biases -= biases_d * learning_step;

	return dx_prev;
}