#include "Layer.h"

Layer::~Layer() {
}

uint32_t Layer::getInputDim() const {
	return _input_shape.size();
}

std::vector<uint32_t> Layer::getInputShape() const {
	return _input_shape;
}

uint32_t Layer::getOutputDim() const {
	return _output_shape.size();
}

std::vector<uint32_t> Layer::getOutputShape() const {
	return _output_shape;
}

void Layer::setNextLayer(Layer* layer) {
	_next_layer = layer;
}

void Layer::setPrevLayer(Layer* layer) {
	_prev_layer = layer;
}

Layer* Layer::getPrevLayer() const {
	return _prev_layer;
}

Layer* Layer::getNextLayer() const {
	return _next_layer;
}

Tensor Layer::getCachedOutput() const {
	return _cached_output;
}