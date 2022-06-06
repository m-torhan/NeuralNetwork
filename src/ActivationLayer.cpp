#include "ActivationLayer.h"

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
	case ActivationFun::LeakyReLU:
		_activation_fun = LeakyReLU_fun;
		_activation_fun_d = LeakyReLU_fun_d;
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
	_cached_output = Tensor(result);
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

void ActivationLayer::summary() const {
	printf("Activation Layer\n");
	printf("  in shape:  (*");
	for (uint32_t i{ 0u }; i < _input_shape.size(); ++i) {
		printf(", %d", _input_shape[i]);
	}
	printf(")  ");
	printf("  out shape: (*");
	for (uint32_t i { 0u }; i < _output_shape.size(); ++i) {
		printf(", %d", _output_shape[i]);
	}
	printf(")\n");
}

uint32_t ActivationLayer::getParamsCount() const {
	return 0;
}

const Tensor ActivationLayer::ReLU_fun(const Tensor& x) {
	return x * (x > 0.0f);
}

const Tensor ActivationLayer::ReLU_fun_d(const Tensor& x, const Tensor& dx) {
	return dx * (x > 0.0f);
}

const Tensor ActivationLayer::LeakyReLU_fun(const Tensor& x) {
	return x.applyFunction([](float value) {return value > 0.0f ? value * 1.0f : value * 0.1f; });
}

const Tensor ActivationLayer::LeakyReLU_fun_d(const Tensor& x, const Tensor& dx) {
	return dx * x.applyFunction([](float value) {return value > 0.0f ? 1.0f : 0.1f; });
}

const Tensor ActivationLayer::Sigmoid_fun(const Tensor& x) {
	return x.applyFunction([](float value) {return powf(1.0f + expf(-value), -1); });
}

const Tensor ActivationLayer::Sigmoid_fun_d(const Tensor& x, const Tensor& dx) {
	Tensor sig = Sigmoid_fun(x);
	return dx * (sig * (1.0f - sig));
}