#pragma once

#include <cstdlib>
#include <cstring>

#include "Utils.h"
#include "Layer.h"

class Conv2DLayer : public Layer {
public:
	/**
	 * @brief Construct a new Conv2D Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param filters_count Convolution filters count.
	 * @param filter_size Convolution filters size.
	 */
	Conv2DLayer(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size);
	/**
	 * @brief Construct a new Conv2D Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param filters_count Convolution filters count.
	 * @param filter_size Convolution filters size.
	 */
	Conv2DLayer(Layer& prev_layer, uint32_t filters_count, uint32_t filter_size);
	
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
	 * Count of the convolution filters.
	 */
	uint32_t _filters_count;
	/**
	 * Size of the convolution filters.
	 */
	uint32_t _filter_size;
	/**
	 * Convolution weights.
	 */
	Tensor _weights;
	/**
	 * Convolution biases.
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
	/**
	 * Cached input slices (used for optimization).
	 */
	Tensor _cached_rects;

	/**
	 * @brief Initializes convolution weights and biases.
	 * 
	 * @param input_shape Shape of the input Tensor.
	 * @param filters_count Convolution filters count.
	 * @param filter_size Convolution filters size.
	 */
	void initWeights(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size);
};