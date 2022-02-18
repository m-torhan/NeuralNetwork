#include "ActivationLayer.h"

#include <cstdlib>
#include <cstring>

ActivationLayer::ActivationLayer( uint32_t input_dim, uint32_t* input_shape, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	initInput(input_dim, input_shape);
	initOutput(input_dim, input_shape);
	initActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(uint32_t input_dim, uint32_t* input_shape, ActivationFun activation_fun) : Layer() {
	initInput(input_dim, input_shape);
	initOutput(input_dim, input_shape);
	initActivationFun(activation_fun);
}

ActivationLayer::ActivationLayer(Layer& prev_layer, Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	initInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initActivationFun(activation_fun, activation_fun_d);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

ActivationLayer::ActivationLayer(Layer& prev_layer, ActivationFun activation_fun) : Layer() {
	initInput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initOutput(prev_layer.getOutputDim(), prev_layer.getOutputShape());
	initActivationFun(activation_fun);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

void ActivationLayer::initActivationFun(Tensor* (*activation_fun)(const Tensor&), Tensor* (*activation_fun_d)(const Tensor&, const Tensor&)) {
	_activation_fun = activation_fun;
	_activation_fun_d = activation_fun_d;
}

void ActivationLayer::initActivationFun(ActivationFun activation_fun) {
	switch (activation_fun) {
	case ActivationFun::Sigmoid:
		_activation_fun = 0;
		_activation_fun_d = 0;
		break;
	case ActivationFun::ReLU:
		_activation_fun = ReLU_fun;
		_activation_fun_d = ReLU_fun_d;
		break;
	default:
		_activation_fun = 0;
		_activation_fun_d = 0;
	}
}

Tensor* ActivationLayer::forwardPropagation(const Tensor& x) {
	Tensor* result;
	_cached_input = new Tensor(x);
	result = _activation_fun(x);
	return result;
}

Tensor* ActivationLayer::backwardPropagation(const Tensor& dx, float learning_step) {
	Tensor* result;
	result = _activation_fun_d(*_cached_input, dx);
	return result;
}

Tensor* ActivationLayer::ReLU_fun(const Tensor& x) {
	return x * (*(x > 0.0f));
}

Tensor* ActivationLayer::ReLU_fun_d(const Tensor& x, const Tensor& dx) {
	return dx * (*(x > 0.0f));
}