#pragma once

#include <cstdlib>
#include <cstring>

#include "Utils.h"
#include "Layer.h"

class Conv2DLayer : public Layer {
public:
	Conv2DLayer(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size);
	Conv2DLayer(Layer& prev_layer, uint32_t filters_count, uint32_t filter_size);
	
	void setWeights(std::vector<float> weights);
	void setBiases(std::vector<float> biases);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum);
	virtual void initCachedGradient();
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	uint32_t _filters_count;
	uint32_t _filter_size;
	Tensor _weights;
	Tensor _biases;
	uint32_t _samples;
	Tensor _cached_weights_d;
	Tensor _cached_weights_d_velocity;
	Tensor _cached_biases_d;
	Tensor _cached_biases_d_velocity;
	Tensor _cached_rects;

	void initWeights(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size);
};