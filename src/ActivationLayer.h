#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "Layer.h"

enum class ActivationFun {
	Sigmoid,
	ReLU,
	LeakyReLU,
	Tanh,
	Softmax
};

class ActivationLayer : public Layer {
public:
	/**
	 * @brief Construct a new Activation Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param activation_fun Activation function.
	 * @param activation_fun_d Activation function derivative.
	 */
	ActivationLayer(std::vector<uint32_t> input_shape, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	/**
	 * @brief Construct a new Activation Layer.
	 * 
	 * @param input_shape Shape of input Tensor.
	 * @param activation_fun Activation function enum.
	 */
	ActivationLayer(std::vector<uint32_t> input_shape, ActivationFun activation_fun);
	/**
	 * @brief Construct a new Activation Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param activation_fun Activation function.
	 * @param activation_fun_d Activation function derivative.
	 */
	ActivationLayer(Layer& prev_layer, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	/**
	 * @brief Construct a new Activation Layer.
	 * 
	 * @param prev_layer Previous layer.
	 * @param activation_fun Activation function enum.
	 */
	ActivationLayer(Layer& prev_layer, ActivationFun activation_fun);

	virtual const Tensor forwardPropagation(const Tensor& x, bool inference=true);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step, float momentum) { };
	virtual void initCachedGradient() { };
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	/**
	 * @brief Initializes activation function.
	 * 
	 * @param activation_fun Activation function.
	 * @param activation_fun_d Activation function derivative.
	 */
	void initActivationFun(const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	/**
	 * @brief Initializes activation function.
	 * 
	 * @param activation_fun Activation function enum.
	 */
	void initActivationFun(ActivationFun activation_fun);

	/**
	 * Activation function.
	 */
	const Tensor (*_activation_fun)(const Tensor&);
	/**
	 * Activation function derivative.
	 */
	const Tensor (*_activation_fun_d)(const Tensor&, const Tensor&);

	/**
	 * Activation functions and their derivatives.
	 */
	static const Tensor ReLU_fun(const Tensor& x);
	static const Tensor ReLU_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor LeakyReLU_fun(const Tensor& x);
	static const Tensor LeakyReLU_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor Sigmoid_fun(const Tensor& x);
	static const Tensor Sigmoid_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor Tanh_fun(const Tensor& x);
	static const Tensor Tanh_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor Softmax_fun(const Tensor& x);
	static const Tensor Softmax_fun_d(const Tensor& x, const Tensor& dx);
};