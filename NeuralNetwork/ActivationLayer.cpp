#include "ActivationLayer.h"

#include <cstdlib>
#include <cstring>

ActivationLayer::ActivationLayer( uint32_t input_dim, uint32_t* input_shape, float (*activation_fun)(float), float (*activation_fun_d)(float, float)) : Layer() {
	InitInput(input_dim, input_shape);
	InitOutput(input_dim, input_shape);
	InitActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(uint32_t input_dim, uint32_t* input_shape, ActivationFun activation_fun) : Layer() {
	InitInput(input_dim, input_shape);
	InitOutput(input_dim, input_shape);
	InitActivationFun(activation_fun);
}

ActivationLayer::ActivationLayer(const Layer& prev_layer, float (*activation_fun)(float), float (*activation_fun_d)(float, float)) : Layer() {
	InitInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(const Layer& prev_layer, ActivationFun activation_fun) : Layer() {
	InitInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitActivationFun(activation_fun);
}

void ActivationLayer::InitActivationFun(float (*activation_fun)(float), float (*activation_fun_d)(float, float)) {
	_activation_fun = activation_fun;
	_activation_fun_d = activation_fun_d;
}

void ActivationLayer::InitActivationFun(ActivationFun activation_fun) {
	switch (activation_fun) {
	case Sigmoid:
		_activation_fun = 0;
		_activation_fun_d = 0;
		break;
	case ReLU:
		_activation_fun = ReLU_fun;
		_activation_fun_d = ReLU_fun_d;
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
	//result->applyFunction(_activation_fun_d);
	return result;
}

float ActivationLayer::ReLU_fun(float x) {
	if (x > 0) {
		return x;
	}
	return 0;
}

float ActivationLayer::ReLU_fun_d(float x, float dx) {
	if (x > 0) {
		return dx;
	}
	return 0;
}