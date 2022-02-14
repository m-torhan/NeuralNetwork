#pragma once
#include "Layer.h"

enum ActivationFun {
	Sigmoid,
	ReLU
};

class ActivationLayer : Layer {
public:
	ActivationLayer(uint32_t input_dim, uint32_t* input_shape, float (*activation_fun)(float), float (*activation_fun_d)(float, float));
	ActivationLayer(uint32_t input_dim, uint32_t* input_shape, ActivationFun activation_fun);
	ActivationLayer(const Layer& prev_layer, float (*activation_fun)(float), float (*activation_fun_d)(float, float));
	ActivationLayer(const Layer& prev_layer, ActivationFun activation_fun);

	virtual Tensor* forwardPropagation(const Tensor& tensor);
	virtual Tensor* backwardPropagation(const Tensor& tensor);

private:
	void InitActivationFun(float (*activation_fun)(float), float (*activation_fun_d)(float, float));
	void InitActivationFun(ActivationFun activation_fun);
	float (*_activation_fun)(float);
	float (*_activation_fun_d)(float, float);

	static float ReLU_fun(float x);
	static float ReLU_fun_d(float x, float dx);
};