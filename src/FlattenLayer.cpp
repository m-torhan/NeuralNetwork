#include "FlattenLayer.h"

FlattenLayer::FlattenLayer(std::vector<uint32_t> input_shape) {
	_input_shape = input_shape;

    uint32_t product = 1;
    for (auto i : input_shape) {
        product *= i;
    }
	_output_shape = { product };
}

FlattenLayer::FlattenLayer(Layer& prev_layer) {
	_input_shape = prev_layer.getOutputShape();

    uint32_t product = 1;
    for (auto i : _input_shape) {
        product *= i;
    }
	_output_shape = { product };
	
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

const Tensor FlattenLayer::forwardPropagation(const Tensor& x, bool inference) {
	auto new_shape = _output_shape;
	new_shape.insert(new_shape.begin(), x.getShape()[0]);
	return x.reshape(new_shape);
}

const Tensor FlattenLayer::backwardPropagation(const Tensor& dx) {
	auto new_shape = _input_shape;
	new_shape.insert(new_shape.begin(), dx.getShape()[0]);
	return dx.reshape(new_shape);
}

void FlattenLayer::summary() const {
	printf("FlattenLayer Layer  ");
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

uint32_t FlattenLayer::getParamsCount() const {
    return 0;
}
