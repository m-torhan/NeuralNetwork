#include "DropoutLayer.h"

DropoutLayer::DropoutLayer(std::vector<uint32_t> input_shape, float rate) : _rate(rate) {
	_input_shape = input_shape;

    uint32_t product = 1;
    for (auto i : input_shape) {
        product *= i;
    }
	_output_shape = { product };
}

DropoutLayer::DropoutLayer(Layer& prev_layer, float rate) : _rate(rate) {
	_input_shape = prev_layer.getOutputShape();

    uint32_t product = 1;
    for (auto i : _input_shape) {
        product *= i;
    }
	_output_shape = { product };
	
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

const Tensor DropoutLayer::forwardPropagation(const Tensor& x, bool inference) {
    Tensor x_next = x;
    if (!inference) {
        Tensor mask = x.applyFunction([] (float) { return (static_cast <float> (rand()) / static_cast <float> (RAND_MAX)); });
        mask = mask > _rate;
        x_next *= mask;

        _cached_input = x;
        _cached_output = x_next;
    }
    
	return x_next;
}

const Tensor DropoutLayer::backwardPropagation(const Tensor& dx) {
	return dx * (_cached_output > 0.0f);
}

void DropoutLayer::summary() const {
	printf("DropoutLayer Layer  ");
	printf("  in shape:  (*");
	for (uint32_t i{ 0u }; i < _input_shape.size(); ++i) {
		printf(", %d", _input_shape[i]);
	}
	printf(")  ");
	printf("  out shape: (*");
	for (uint32_t i { 0u }; i < _output_shape.size(); ++i) {
		printf(", %d", _output_shape[i]);
	}
	printf(")  rate: %.3f\n", _rate);
}

uint32_t DropoutLayer::getParamsCount() const {
    return 0;
}
