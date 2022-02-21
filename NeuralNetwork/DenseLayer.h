#pragma once
#include "Layer.h"


class DenseLayer : public Layer {
public:
	DenseLayer(std::vector<uint32_t> input_shape, uint32_t neurons_count);
	DenseLayer(Layer& prev_layer, uint32_t neurons_count);
	
	virtual const Tensor forwardPropagation(const Tensor& x);
	virtual const Tensor backwardPropagation(const Tensor& dx, float learning_step);
	virtual void updateWeights();
	virtual void initCachedGradient(); 

private:
	uint32_t _neurons_count;
	Tensor _weights;
	Tensor _biases;
	uint32_t _samples;
	Tensor _cached_weights_d;
	Tensor _cached_biases_d;

	void initWeights(std::vector<uint32_t> input_shape, uint32_t neurons_count);
};