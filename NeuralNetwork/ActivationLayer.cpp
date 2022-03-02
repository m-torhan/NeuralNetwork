#include "ActivationLayer.h"

#include <cstdlib>
#include <cstring>
#include <cmath>

ActivationLayer::ActivationLayer(std::vector<uint32_t> input_shape, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	_input_shape = input_shape;
	_output_shape = input_shape;
	initActivationFun(activation_fun, activation_fun_d);
}

ActivationLayer::ActivationLayer(std::vector<uint32_t> input_shape, ActivationFun activation_fun) : Layer() {
	_input_shape = input_shape;
	_output_shape = input_shape;
	initActivationFun(activation_fun);
}

ActivationLayer::ActivationLayer(Layer& prev_layer, const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&)) : Layer() {
	_input_shape = prev_layer.getOutputShape();
	_output_shape = prev_layer.getOutputShape();
	initActivationFun(activation_fun, activation_fun_d);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

ActivationLayer::ActivationLayer(Layer& prev_layer, ActivationFun activation_fun) : Layer() {
	_input_shape = prev_layer.getOutputShape();
	_output_shape = prev_layer.getOutputShape();
	initActivationFun(activation_fun);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

void ActivationLayer::initActivationFun(const Tensor (*activation_fun)(const Tensor&), const Tensor (*activation_fun_d)(const Tensor&, const Tensor&)) {
	_activation_fun = activation_fun;
	_activation_fun_d = activation_fun_d;
}

void ActivationLayer::initActivationFun(ActivationFun activation_fun) {
	switch (activation_fun) {
	case ActivationFun::Sigmoid:
		_activation_fun = Sigmoid_fun;
		_activation_fun_d = Sigmoid_fun_d;
		break;
	case ActivationFun::ReLU:
		_activation_fun = ReLU_fun;
		_activation_fun_d = ReLU_fun_d;
		break;
	default:
		_activation_fun = nullptr;
		_activation_fun_d = nullptr;
		// exception
	}
}

const Tensor ActivationLayer::forwardPropagation(const Tensor& x) {
	_cached_input = Tensor(x);
	Tensor result = _activation_fun(x);
	return result;
}

const Tensor ActivationLayer::backwardPropagation(const Tensor& dx) {
	Tensor result = _activation_fun_d(_cached_input, dx);
	return result;
}

void ActivationLayer::updateWeights(float learning_step) {

}

void ActivationLayer::initCachedGradient() {

}

const Tensor ActivationLayer::ReLU_fun(const Tensor& x) {
	return x * (x > 0.0f);
}

const Tensor ActivationLayer::ReLU_fun_d(const Tensor& x, const Tensor& dx) {
	return dx * (x > 0.0f);
}

const Tensor ActivationLayer::Sigmoid_fun(const Tensor& x) {
	Tensor result;
	result = x.applyFunction([](float value) {return powf(1.0f + expf(-value), -1); });
	return result;
}

const Tensor ActivationLayer::Sigmoid_fun_d(const Tensor& x, const Tensor& dx) {
	Tensor sig = Sigmoid_fun(x);
	return dx * (sig * (-sig + 1.0f));
}