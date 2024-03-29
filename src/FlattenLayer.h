#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "Layer.h"

class FlattenLayer : public Layer {
public:
	/**
	 * @brief Construct a new Flatten Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 */
	FlattenLayer(std::vector<uint32_t> input_shape);
	/**
	 * @brief Construct a new Flatten Layer.
	 * 
	 * @param prev_layer Previous layer.
	 */
	FlattenLayer(Layer& prev_layer);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) { };
	virtual void initCachedGradient() { };
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

};