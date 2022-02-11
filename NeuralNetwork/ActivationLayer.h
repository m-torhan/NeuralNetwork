#pragma once
#include "Layer.h"

enum ActivationFun {
	Sigmoid,
	ReLU
};

class ActivationLayer : Layer {
public:
	ActivationLayer(float (*activation_fun)(float), float (*activation_fun_d)(float));
	ActivationLayer(ActivationFun activation_fun);

	virtual Tensor* forwardPropagation(const Tensor& tensor);
	virtual Tensor* backwardPropagation(const Tensor& tensor);
private:
	float (*_activation_fun)(float);
	float (*_activation_fun_d)(float);
};