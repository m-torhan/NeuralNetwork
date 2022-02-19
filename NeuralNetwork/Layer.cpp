#include "Layer.h"

#include <cstdlib>
#include <cstring>

uint32_t Layer::getInputDim() const {
	return _input_dim;
}

uint32_t* Layer::getInputShape() const {
	uint32_t* input_shape;

	input_shape = (uint32_t*)malloc(sizeof(uint32_t) * _input_dim);
	memcpy(input_shape, _input_shape, sizeof(uint32_t) * _input_dim);

	return input_shape;
}

uint32_t Layer::getOutputDim() const {
	return _output_dim;
}

uint32_t* Layer::getOutputShape() const {
	uint32_t* output_shape;

	output_shape = (uint32_t*)malloc(sizeof(uint32_t) * _output_dim);
	memcpy(output_shape, _output_shape, sizeof(uint32_t) * _output_dim);

	return output_shape;
}

void Layer::initInput(uint32_t input_dim, uint32_t* input_shape) {
	_input_dim = input_dim;
	_input_shape = (uint32_t*)malloc(sizeof(uint32_t) * _input_dim);
	if (!_input_shape) {
		// exception
	}
	memcpy(_input_shape, input_shape, sizeof(uint32_t) * _input_dim);
}

void Layer::initOutput(uint32_t output_dim, uint32_t* output_shape) {
	_output_dim = output_dim;
	_output_shape = (uint32_t*)malloc(sizeof(uint32_t) * _output_dim);
	if (!_output_shape) {
		// exception
	}
	memcpy(_output_shape, output_shape, sizeof(uint32_t) * _output_dim);
}

void Layer::setNextLayer(Layer* layer) {
	_next_layer = layer;
}

void Layer::setPrevLayer(Layer* layer) {
	_prev_layer = layer;
}

Layer* Layer::getPrevLayer() const {
	return _prev_layer;
}

Layer* Layer::getNextLayer() const {
	return _next_layer;
}