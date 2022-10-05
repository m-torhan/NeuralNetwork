#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "Layer.h"

class ReshapeLayer : public Layer {
public:
	/**
	 * @brief Construct a new Reshape Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param output_shape Shape of output Tensor.
	 */
	ReshapeLayer(std::vector<uint32_t> input_shape, std::vector<uint32_t> output_shape);
	/**
	 * @brief Construct a new Reshape Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param output_shape Shape of output Tensor.
	 */
	ReshapeLayer(Layer& prev_layer, std::vector<uint32_t> output_shape);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) { };
	virtual void initCachedGradient() { };
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

};