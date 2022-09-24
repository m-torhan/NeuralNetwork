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
	case ActivationFun::ReLU:
		_activation_fun = ReLU_fun;
		_activation_fun_d = ReLU_fun_d;
		break;
	case ActivationFun::LeakyReLU:
		_activation_fun = LeakyReLU_fun;
		_activation_fun_d = LeakyReLU_fun_d;
		break;
	case ActivationFun::Sigmoid:
		_activation_fun = Sigmoid_fun;
		_activation_fun_d = Sigmoid_fun_d;
		break;
	case ActivationFun::Tanh:
		_activation_fun = Tanh_fun;
		_activation_fun_d = Tanh_fun_d;
		break;
	case ActivationFun::Softmax:
		_activation_fun = Softmax_fun;
		_activation_fun_d = Softmax_fun_d;
		break;
	default:
		_activation_fun = nullptr;
		_activation_fun_d = nullptr;
		throw std::invalid_argument(format_string("%s %d : Provided activation function in invalid. Provided value: %d",
			__FILE__, __LINE__, activation_fun));
	}
}

const Tensor ActivationLayer::forwardPropagation(const Tensor& x, bool inference) {
	Tensor result = _activation_fun(x);
	if (!inference) {
		_cached_input = x;
		_cached_output = result;
	}
	return result;
}

const Tensor ActivationLayer::backwardPropagation(const Tensor& dx) {
	Tensor result = _activation_fun_d(_cached_input, dx);
	return result;
}

void ActivationLayer::summary() const {
	printf("Activation Layer    ");
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

const Tensor ActivationLayer::Tanh_fun(const Tensor& x) {
	return x.applyFunction(tanhf);
}

const Tensor ActivationLayer::Tanh_fun_d(const Tensor& x, const Tensor& dx) {
	Tensor t = std::move(x.applyFunction(tanhf));
	return dx * (1 - (t * t));
}

const Tensor ActivationLayer::Softmax_fun(const Tensor& x) {
	Tensor x_next(x.getShape());
	
	for (uint32_t i{ 0 }; i < x.getShape()[0]; ++i) {
		Tensor x_row = x[{ { i }, { 0, x.getShape()[1] }}];
		Tensor exp_x_row = (x_row - x_row.max()).applyFunction(expf);
		x_next[{ { i }, { 0, x.getShape()[1] }}] = exp_x_row / exp_x_row.sum();
	}
	
	return x_next;
}

const Tensor ActivationLayer::Softmax_fun_d(const Tensor& x, const Tensor& dx) {
	Tensor dx_prev(x.getShape());

	for (uint32_t i{ 0 }; i < x.getShape()[0]; ++i) {
		for (uint32_t j{ 0 }; j < x.getShape()[1]; ++j) {
			for (uint32_t k{ 0 }; k < x.getShape()[1]; ++k) {
				if (j == k) {
					dx_prev[{{ i }, { j }}] = dx[{{ i }, { j }}] * (1.0f - dx[{{ i }, { k }}]);
				}
				else {
					dx_prev[{{ i }, { j }}] = - dx[{{ i }, { j }}] * dx[{{ i }, { k }}];
				}
			}
		}
	}

	return dx_prev;
}