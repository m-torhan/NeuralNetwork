#include "ActivationLayer.h"

#include <cstdlib>
#include <cstring>

ActivationLayer::ActivationLayer( uint32_t input_dim, uint32_t* input_shape, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	InitInput(input_dim, input_shape);
	InitOutput(input_dim, input_shape);
	InitActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(uint32_t input_dim, uint32_t* input_shape, ActivationFun activation_fun) : Layer() {
	InitInput(input_dim, input_shape);
	InitOutput(input_dim, input_shape);
	InitActivationFun(activation_fun);
}

ActivationLayer::ActivationLayer(const Layer& prev_layer, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	InitInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(const Layer& prev_layer, ActivationFun activation_fun) : Layer() {
	InitInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	InitActivationFun(activation_fun);
}

void ActivationLayer::InitActivationFun(Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) {
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
	Tensor* result;
	result = _activation_fun(tensor);
	_cached_output = new Tensor(*result);
	return result;
}

Tensor* ActivationLayer::backwardPropagation(const Tensor& tensor) {
	Tensor* result;
	result = _activation_fun_d(*_cached_output, tensor);
	return result;
}

Tensor* ActivationLayer::ReLU_fun(const Tensor& x) {
	return x * (*(x > 0.0f));
}

Tensor* ActivationLayer::ReLU_fun_d(const Tensor& x, const Tensor& dx) {
	return dx * (*(x > 0.0f));
}