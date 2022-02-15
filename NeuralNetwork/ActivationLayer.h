#pragma once
#include "Layer.h"

enum ActivationFun {
	Sigmoid,
	ReLU
};

class ActivationLayer : Layer {
public:
	ActivationLayer(uint32_t input_dim, uint32_t* input_shape, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&));
	ActivationLayer(uint32_t input_dim, uint32_t* input_shape, ActivationFun activation_fun);
	ActivationLayer(const Layer& prev_layer, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&));
	ActivationLayer(const Layer& prev_layer, ActivationFun activation_fun);

	virtual Tensor* forwardPropagation(const Tensor& tensor);
	virtual Tensor* backwardPropagation(const Tensor& tensor);

private:
	void InitActivationFun(Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&));
	void InitActivationFun(ActivationFun activation_fun);
	Tensor* (*_activation_fun)(const Tensor&);
	Tensor* (*_activation_fun_d)(const Tensor&, const Tensor&);

	static Tensor* ReLU_fun(const Tensor& x);
	static Tensor* ReLU_fun_d(const Tensor& x, const Tensor& dx);
};