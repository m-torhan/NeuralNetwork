#pragma once

#include <cstdlib>
#include <cstring>

#include "Utils.h"
#include "Layer.h"

class DenseLayer : public Layer {
public:
	/**
	 * @brief Construct a new Dense Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param neurons_count Neurons count in layer.
	 */
	DenseLayer(std::vector<uint32_t> input_shape, uint32_t neurons_count);
	/**
	 * @brief Construct a new Dense Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param neurons_count Neurons count in layer.
	 */
	DenseLayer(Layer& prev_layer, uint32_t neurons_count);
	
	/**
	 * @brief Set the layer weights.
	 * 
	 * @param weights Weights values to be set.
	 */
	void setWeights(std::vector<float> weights);
	/**
	 * @brief Set the layer biases.
	 * 
	 * @param biases Biases values to be set.
	 */
	void setBiases(std::vector<float> biases);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum);
	virtual void initCachedGradient();
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	/**
	 * Neurons count in Layer.
	 */
	uint32_t _neurons_count;
	/**
	 * Dense Layer weights.
	 */
	Tensor _weights;
	/**
	 * Dense Layer biases.
	 */
	Tensor _biases;
	/**
	 * Number of cached samples.
	 */
	uint32_t _samples;
	/**
	 * Cached gradient of weights.
	 */
	Tensor _cached_weights_d;
	/**
	 * Cached gradient velocity of weights.
	 */
	Tensor _cached_weights_d_velocity;
	/**
	 * Cached biases of weights.
	 */
	Tensor _cached_biases_d;
	/**
	 * Cached biases velocity of weights.
	 */
	Tensor _cached_biases_d_velocity;

	void initWeights(std::vector<uint32_t> input_shape, uint32_t neurons_count);
};