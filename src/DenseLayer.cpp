#include "DenseLayer.h"

DenseLayer::DenseLayer(std::vector<uint32_t> input_shape, uint32_t neurons_count) : Layer() {
	_input_shape = input_shape;
	_output_shape = { neurons_count };
	initWeights(_input_shape, neurons_count);
}

DenseLayer::DenseLayer(Layer& prev_layer, uint32_t neurons_count) : Layer() {
	_input_shape = prev_layer.getOutputShape();
	_output_shape = { neurons_count };
	initWeights(_input_shape, neurons_count);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

void DenseLayer::setWeights(std::vector<float> weights) {
	_weights.setValues(weights);
}

void DenseLayer::setBiases(std::vector<float> biases) {
	_biases.setValues(biases);
}

void DenseLayer::initWeights(std::vector<uint32_t> input_shape, uint32_t neurons_count) {
	_neurons_count = neurons_count;

	uint32_t input_size{ 1 };
	for (auto i : input_shape) {
		input_size *= i;
	}

	std::vector<uint32_t> shape = { _neurons_count, input_size };

	_weights = Tensor(shape);

	for (uint32_t i{ 0 }; i < _neurons_count; ++i) {
		for (uint32_t j{ 0 }; j < input_size; ++j) {
			_weights[{ i, j }] = randUniform(-1.0f, 1.0f)*sqrtf(6.0f/input_size);
		}
	}

	shape.pop_back();

	_biases = Tensor(shape);
	_biases *= 0.0f;

	_cached_weights_d_velocity = Tensor(_weights.getShape());
	_cached_biases_d_velocity = Tensor(_biases.getShape());
}

void DenseLayer::initCachedGradient() {
	_cached_weights_d = Tensor(_weights);
	_cached_biases_d = Tensor(_biases);
	_cached_weights_d *= 0.0f;
	_cached_biases_d *= 0.0f;
	_samples = 0;
}

void DenseLayer::summary() const {
	printf("Dense Layer         ");
	printf("  in shape:  (*");
	for (uint32_t i{ 0u }; i < _input_shape.size(); ++i) {
		printf(", %d", _input_shape[i]);
	}
	printf(")  ");
	printf("  out shape: (*");
	for (uint32_t i { 0u }; i < _output_shape.size(); ++i) {
		printf(", %d", _output_shape[i]);
	}
	printf(")  total params: %d\n", _weights.getSize() + _biases.getSize());
}

uint32_t DenseLayer::getParamsCount() const {
	return _weights.getSize() + _biases.getSize();
}

void DenseLayer::updateWeights(float learning_step, float momentum) {
	_cached_weights_d_velocity = (_cached_weights_d * learning_step / _samples) + momentum * _cached_weights_d_velocity;
	_cached_biases_d_velocity = (_cached_biases_d * learning_step / _samples) + momentum * _cached_biases_d_velocity;

	_weights -= _cached_weights_d_velocity;
	_biases -= _cached_biases_d_velocity;
}

const Tensor DenseLayer::forwardPropagation(const Tensor& x, bool inference) {
	Tensor x_flatten;
	if (x.getDim() > 2) {
		x_flatten = x.flatten(1);
	}
	else {
		x_flatten = x;
	}
	Tensor x_next = _weights.dotProductTranspose(x_flatten).transpose() + _biases;
	if (!inference) {
		_cached_input = x;
		_cached_output = x_next;
	}
	return x_next;
}

const Tensor DenseLayer::backwardPropagation(const Tensor& dx) {
	uint32_t n;
	Tensor dx_prev = Tensor(dx);

	n = _cached_input.getShape()[0];
	_samples += n;

	Tensor weights_d = dx.transpose().dotProductTranspose(_cached_input.transpose());
	Tensor biases_d = dx.sum(0);

	dx_prev = dx.dotProductTranspose(_weights.transpose());

	_cached_weights_d += weights_d;
	_cached_biases_d += biases_d;

	return dx_prev;
}