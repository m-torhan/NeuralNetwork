#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "Utils.h"
#include "Tensor.h"

class Layer {
public:
	/**
	 * @brief Get the input dimension.
	 * 
	 * @return Dim of input tensor. 
	 */
	uint32_t getInputDim() const;
	/**
	 * @brief Get the input shape.
	 * 
	 * @return shape of input tensor.
	 */
	std::vector<uint32_t> getInputShape() const;
	/**
	 * @brief Get the output dimension.
	 * 
	 * @return Dim of output tensor. 
	 */
	uint32_t getOutputDim() const;
	/**
	 * @brief Get the output shape.
	 * 
	 * @return shape of output tensor.
	 */
	std::vector<uint32_t> getOutputShape() const;
	/**
	 * @brief Set the previous layer.
	 * 
	 * @param layer Another layer.
	 */
	void setPrevLayer(Layer* layer);
	/**
	 * @brief Set the next layer.
	 * 
	 * @param layer Another layer.
	 */
	void setNextLayer(Layer* layer);
	/**
	 * @brief Get the previous Layer.
	 * 
	 * @return Pointer to previous layer.
	 */
	Layer* getPrevLayer() const;
	/**
	 * @brief Get the next Layer.
	 * 
	 * @return Pointer to next layer.
	 */
	Layer* getNextLayer() const;
	/**
	 * @brief Get the last layer output.
	 * 
	 * @return Returns cached output.
	 */
	Tensor getCachedOutput() const;

	/**
	 * @brief Forward propagation of the layer.
	 * 
	 * @param x Input tensor.
	 * @param inference Inference indicator.
	 * @return Output of the layer for given x.
	 */
	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true) = 0;
	/**
	 * @brief Backward propagation of the layer.
	 * 
	 * @param dx Gradient for the layer output.
	 * @return Gradient for the input.
	 */
	virtual const Tensor backwardPropagation(const Tensor& dx) = 0;
	/**
	 * @brief Updates layer params according to gradient.
	 * 
	 * @param learning_step The learnign step.
	 * @param momentum Momentum used to compute gradient velocity.
	 */
	virtual void updateWeights(float learning_step, float momentum) = 0;
	/**
	 * @brief Initializes cached gradient.
	 * 
	 */
	virtual void initCachedGradient() = 0;
	/**
	 * @brief Print layer summary.
	 * 
	 */
	virtual void summary() const = 0;
	/**
	 * @brief Get the param count.
	 * 
	 * @return Param counf of the layer.
	 */
	virtual uint32_t getParamsCount() const = 0;

protected:
	/**
	 * The layer which input is connected to this layer output.
	 */
	Layer* _next_layer;
	/**
	 * The layer which output is connected to this layer input.
	 */
	Layer* _prev_layer;

	/**
	 * Input tensor shape.
	 */
	std::vector<uint32_t> _input_shape;
	/**
	 * Output tensor shape.
	 */
	std::vector<uint32_t> _output_shape;

	/**
	 * Most recent layer input.
	 */
	Tensor _cached_input;
	/**
	 * Most recent layer output.
	 */
	Tensor _cached_output;
};