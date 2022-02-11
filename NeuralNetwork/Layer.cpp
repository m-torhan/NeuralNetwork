#include "Layer.h"

#include <cstdlib>
#include <cstring>

uint32_t Layer::getInputDim() {
	return _input_dim;
}

uint32_t* Layer::getInputShape() {
	uint32_t* input_shape;

	input_shape = (uint32_t*)malloc(sizeof(uint32_t) * _input_dim);
	memcpy(input_shape, _input_shape, sizeof(uint32_t) * _input_dim);

	return input_shape;
}

uint32_t Layer::getOutputDim() {
	return _output_dim;
}

uint32_t* Layer::getOutputShape() {
	uint32_t* output_shape;

	output_shape = (uint32_t*)malloc(sizeof(uint32_t) * _output_dim);
	memcpy(output_shape, _output_shape, sizeof(uint32_t) * _output_dim);

	return output_shape;
}