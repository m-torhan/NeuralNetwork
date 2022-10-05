#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <random>

#include "Layer.h"

class NormalDistLayer : public Layer {
public:
	/**
	 * @brief Construct a new Normal Distribution Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 */
	NormalDistLayer(std::vector<uint32_t> input_shape);
	/**
	 * @brief Construct a new Normal Distribution Layer.
	 * 
	 * @param prev_layer Previous layer.
	 */
	NormalDistLayer(Layer& prev_layer);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) {};
	virtual void initCachedGradient() {};
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	/**
	 * Random tensor used in most recent forward propagation.
	 */
    Tensor _cached_random_tensor;
};