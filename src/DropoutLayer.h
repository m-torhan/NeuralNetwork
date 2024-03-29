#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "Layer.h"

class DropoutLayer : public Layer {
public:
	/**
	 * @brief Construct a new Droput Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param rate Dropout rate.
	 */
	DropoutLayer(std::vector<uint32_t> input_shape, float rate);
	/**
	 * @brief Construct a new Droput Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param rate Dropout rate.
	 */
	DropoutLayer(Layer& prev_layer, float rate);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) { };
	virtual void initCachedGradient() { };
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	/**
	 * Dropout rate.
	 */
    float _rate;
};