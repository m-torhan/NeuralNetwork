#include "ReshapeLayer.h"

ReshapeLayer::ReshapeLayer(std::vector<uint32_t> input_shape, std::vector<uint32_t> output_shape) {
	_input_shape = input_shape;
	_output_shape = output_shape;
}

ReshapeLayer::ReshapeLayer(Layer& prev_layer, std::vector<uint32_t> output_shape) {
	_input_shape = prev_layer.getOutputShape();
	_output_shape = output_shape;
	
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

const Tensor ReshapeLayer::forwardPropagation(const Tensor& x) {
	auto new_shape = _output_shape;
	new_shape.insert(new_shape.begin(), x.getShape()[0]);
	return x.reshape(new_shape);
}

const Tensor ReshapeLayer::backwardPropagation(const Tensor& dx) {
	auto new_shape = _input_shape;
	new_shape.insert(new_shape.begin(), dx.getShape()[0]);
	return dx.reshape(new_shape);
}

void ReshapeLayer::updateWeights(float learning_step) {}

void ReshapeLayer::initCachedGradient() {}

void ReshapeLayer::summary() const {
	printf("ReshapeLayer Layer  ");
	printf("  in shape:  (*");
	for (uint32_t i{ 0u }; i < _input_shape.size(); ++i) {
		printf(", %d", _input_shape[i]);
	}
	printf(")  ");
	printf("  out shape: (*");
	for (uint32_t i { 0u }; i < _output_shape.size(); ++i) {
		printf(", %d", _output_shape[i]);
	}
	printf(")  total params: %d\n", getParamsCount());
}

uint32_t ReshapeLayer::getParamsCount() const {
    return 0;
}
