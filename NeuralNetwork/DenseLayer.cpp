#include "DenseLayer.h"

#include <cstdlib>
#include <cstring>
#include <random>

DenseLayer::DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count) {
	initInput(input_dim, input_shape);
	initWeights(input_dim, input_shape, neurons_count);
}

DenseLayer::DenseLayer(Layer& prev_layer, uint32_t neurons_count) {
	initInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initWeights(prev_layer.getOutputDim(), prev_layer.getOutputShape(), neurons_count);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

void DenseLayer::initWeights(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count) {
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t input_size = 1;
	uint32_t* shape;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<float> d(0, 1);

	_neurons_count = neurons_count;

	for (i = 0; i < input_dim; ++i) {
		input_size *= input_shape[i];
	}

	shape = (uint32_t*)malloc(sizeof(uint32_t) * 2);
	if (!shape) {
		// exception
	}

	shape[0] = _neurons_count;
	shape[1] = input_size;

	_weights = new Tensor(2, shape);

	for (i = 0; i < _neurons_count; ++i) {
		for (j = 0; j < input_size; ++j) {
			_weights->setValue(d(gen), 2, i, j);
		}
	}

	shape = (uint32_t*)realloc(shape, sizeof(uint32_t));
	if (!shape) {
		// exception
	}

	shape[0] = _neurons_count;

	_biases = new Tensor(1, shape);

	for (i = 0; i < _neurons_count; ++i) {
		_biases->setValue(d(gen), 1, i);
	}

	free(shape);
}

Tensor* DenseLayer::forwardPropagation(const Tensor& x) {
	Tensor* x_next = x.flatten(1);
	_cached_input = new Tensor(*x_next);
	x_next = (*_weights->dotProduct(*x_next->transpose()) + *_biases)->transpose();
	return x_next;
}

Tensor* DenseLayer::backwardPropagation(const Tensor& dx, float learning_step) {
	uint32_t n;
	Tensor* dx_prev = new Tensor(dx);
	Tensor* weights_d;
	Tensor* biases_d;

	n = _cached_input->getShape()[0];

	weights_d = *dx.transpose()->dotProduct(*_cached_input) / n;
	biases_d = *dx.sum(0) / n;

	dx_prev = dx.dotProduct(*_weights);

	*_weights -= *((*weights_d) * learning_step);
	*_biases -= *((*biases_d) * learning_step);

	return dx_prev;
}