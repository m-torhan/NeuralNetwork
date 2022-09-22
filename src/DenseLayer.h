#pragma once

#include <cstdlib>
#include <cstring>

#include "Utils.h"
#include "Layer.h"

class DenseLayer : public Layer {
public:
	DenseLayer(std::vector<uint32_t> input_shape, uint32_t neurons_count);
	DenseLayer(Layer& prev_layer, uint32_t neurons_count);
	
	void setWeights(std::vector<float> weights);
	void setBiases(std::vector<float> biases);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum);
	virtual void initCachedGradient();
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	uint32_t _neurons_count;
	Tensor _weights;
	Tensor _biases;
	uint32_t _samples;
	Tensor _cached_weights_d;
	Tensor _cached_weights_d_velocity;
	Tensor _cached_biases_d;
	Tensor _cached_biases_d_velocity;

	void initWeights(std::vector<uint32_t> input_shape, uint32_t neurons_count);
};