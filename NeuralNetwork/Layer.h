#pragma once
#include "Tensor.h"

#include <cstdint>

class Layer {
public:
	uint32_t getInputDim();
	uint32_t* getInputShape();
	uint32_t getOutputDim();
	uint32_t* getOutputShape();
	virtual Tensor* forwardPropagation(const Tensor& tensor) = 0;
	virtual Tensor* backwardPropagation(const Tensor& tensor) = 0;

protected:
	uint32_t _input_dim;
	uint32_t* _input_shape;
	uint32_t _output_dim;
	uint32_t* _output_shape;
};