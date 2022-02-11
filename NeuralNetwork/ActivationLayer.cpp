#include "ActivationLayer.h"

ActivationLayer::ActivationLayer(float (*activation_fun)(float), float (*activation_fun_d)(float)) {
	_activation_fun = activation_fun;
	_activation_fun_d = activation_fun_d;
}

ActivationLayer::ActivationLayer(ActivationFun activation_fun) {
	switch (activation_fun) {
	case Sigmoid:
		break;
	case ReLU:
		break;
	default:
		_activation_fun = 0;
		_activation_fun_d = 0;
	}
}

Tensor* ActivationLayer::forwardPropagation(const Tensor& tensor) {
	Tensor* result = new Tensor(tensor);
	result->applyFunction(_activation_fun);
	return result;
}

Tensor* ActivationLayer::backwardPropagation(const Tensor& tensor) {
	Tensor* result = new Tensor(tensor);
	result->applyFunction(_activation_fun_d);
	return result;
}