#include "DenseLayer.h"

#include <cstdlib>
#include <cstring>
#include <random>

DenseLayer::DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count) {
	initInput(input_dim, input_shape);
	initOutput(1, &neurons_count);
	initWeights(input_dim, input_shape, neurons_count);
}

DenseLayer::DenseLayer(Layer& prev_layer, uint32_t neurons_count) {
	initInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initOutput(1, &neurons_count);
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
			_weights->setValue(d(gen) * 0.1f, 2, i, j);
		}
	}

	shape = (uint32_t*)realloc(shape, sizeof(uint32_t));
	if (!shape) {
		// exception
	}

	shape[0] = _neurons_count;

	_biases = new Tensor(1, shape);

	for (i = 0; i < _neurons_count; ++i) {
		_biases->setValue(d(gen) * 0.1f, 1, i);
	}

	free(shape);
}

void DenseLayer::initCachedGradient() {
	_cached_weights_d = new Tensor(*_weights);
	_cached_biases_d = new Tensor(*_biases);
	*_cached_weights_d *= 0.0f;
	*_cached_biases_d *= 0.0f;
	_samples = 0;
}

void DenseLayer::updateWeights() {
	*_weights -= *(*_cached_weights_d / _samples);
	*_biases -= *(*_cached_biases_d / _samples);
}

Tensor* DenseLayer::forwardPropagation(const Tensor& x) {
	Tensor* x_next;
	if (x.getDim() > 2) {
		x_next = x.flatten(1);
	}
	else {
		x_next = new Tensor(x);
	}
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
	_samples += n;

	weights_d = dx.transpose()->dotProduct(*_cached_input);
	biases_d = dx.sum(0);

	dx_prev = dx.dotProduct(*_weights);

	*_cached_weights_d += *((*weights_d) * learning_step);
	*_cached_biases_d += *((*biases_d) * learning_step);

	return dx_prev;
}