#pragma once
#include "Layer.h"


class DenseLayer : Layer {
public:
	DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count);
	DenseLayer(const Layer& prev_layer, uint32_t neurons_count);
	
	virtual Tensor* forwardPropagation(const Tensor& x);
	virtual Tensor* backwardPropagation(const Tensor& dx);

private:
	uint32_t _neurons_count;
	Tensor* _weights;
	Tensor* _biases;

	void InitWeights(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count);
};