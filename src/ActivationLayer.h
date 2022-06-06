#pragma once

#include <cstdlib>
#include <cstring>
#include <cmath>

#include "Layer.h"

enum class ActivationFun {
	Sigmoid,
	ReLU,
	LeakyReLU
};

class ActivationLayer : public Layer {
public:
	ActivationLayer(std::vector<uint32_t> input_shape, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	ActivationLayer(std::vector<uint32_t> input_shape, ActivationFun activation_fun);
	ActivationLayer(Layer& prev_layer, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	ActivationLayer(Layer& prev_layer, ActivationFun activation_fun);

	virtual const Tensor forwardPropagation(const Tensor& x);
	virtual const Tensor backwardPropagation(const Tensor& dx);
	virtual void updateWeights(float learning_step);
	virtual void initCachedGradient();
	virtual void summary() const;
	virtual uint32_t getParamsCount() const;

private:
	void initActivationFun(const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&));
	void initActivationFun(ActivationFun activation_fun);
	const Tensor (*_activation_fun)(const Tensor&);
	const Tensor (*_activation_fun_d)(const Tensor&, const Tensor&);

	static const Tensor ReLU_fun(const Tensor& x);
	static const Tensor ReLU_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor LeakyReLU_fun(const Tensor& x);
	static const Tensor LeakyReLU_fun_d(const Tensor& x, const Tensor& dx);
	static const Tensor Sigmoid_fun(const Tensor& x);
	static const Tensor Sigmoid_fun_d(const Tensor& x, const Tensor& dx);
};