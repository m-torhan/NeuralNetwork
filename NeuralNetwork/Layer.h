#pragma once
#include "Tensor.h"

#include <cstdint>

class Layer {
public:
	uint32_t getInputDim() const;
	uint32_t* getInputShape() const;
	uint32_t getOutputDim() const;
	uint32_t* getOutputShape() const;

	virtual Tensor* forwardPropagation(const Tensor& tensor) = 0;
	virtual Tensor* backwardPropagation(const Tensor& tensor) = 0;

protected:
	uint32_t _input_dim;
	uint32_t* _input_shape;
	uint32_t _output_dim;
	uint32_t* _output_shape;

	void InitInput(uint32_t input_dim, uint32_t* input_shape);
	void InitOutput(uint32_t output_dim, uint32_t* output_shape);
};