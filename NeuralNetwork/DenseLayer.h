#pragma once
#include "Layer.h"


class DenseLayer : Layer {
public:
	DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count);
	
	virtual Tensor* forwardPropagation(const Tensor& tensor);
	virtual Tensor* backwardPropagation(const Tensor& tensor);

private:
	uint32_t _neurons_count;
	float* weights;
	float* bias;
};