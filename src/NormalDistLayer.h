#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

#include "Layer.h"

class NormalDistLayer : public Layer {
public:
	NormalDistLayer(std::vector<uint32_t> input_shape);
	NormalDistLayer(Layer& prev_layer);

	virtual const Tensor forwardPropagation(const Tensor& x);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step);
	virtual void initCachedGradient();
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
    Tensor _cached_random_tensor;
};