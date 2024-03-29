#pragma once

#include <cstdlib>
#include <cstring>

#include "Utils.h"
#include "Layer.h"

enum class PoolMode {
	Max,
    Average
};

class Pool2DLayer : public Layer {
public:
	/**
	 * @brief Construct a new Pool 2D Layer.
	 * 
	 * @param input_shape Size of the input Tensor.
	 * @param pool_size Size of pooling square.
	 * @param pool_mode Pooling function used.
	 */
	Pool2DLayer(std::vector<uint32_t> input_shape, int32_t pool_size, PoolMode pool_mode);
	/**
	 * @brief Construct a new Pool 2D Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param pool_size Size of pooling square.
	 * @param pool_mode Pooling function used.
	 */
	Pool2DLayer(Layer& prev_layer, int32_t pool_size, PoolMode pool_mode);
	
	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) { };
	virtual void initCachedGradient() { };
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	/**
	 * Size of the pooling square.
	 */
	uint32_t _pool_size;
	/**
	 * Function used in pooling.
	 */
	float (*_pool_function)(const Tensor& x);
	/**
	 * Pooling function derivative.
	 */
	const Tensor (*_pool_function_d)(const Tensor& x, float dx);

	/**
	 * Pooling functions and their derivatives.
	 */
	static float pool_max(const Tensor& x);
	static const Tensor pool_max_d(const Tensor& x, float dx);
	static float pool_mean(const Tensor& x);
	static const Tensor pool_mean_d(const Tensor& x, float dx);
};