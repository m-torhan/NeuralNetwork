#pragma once
#include "Tensor.h"

#include <cstdint>

class Layer {
public:
	uint32_t getInputDim() const;
	uint32_t* getInputShape() const;
	uint32_t getOutputDim() const;
	uint32_t* getOutputShape() const;

	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	Layer* getNextLayer();
	Layer* setNextLayer();

	virtual Tensor* forwardPropagation(const Tensor& x) = 0;
	virtual Tensor* backwardPropagation(const Tensor& dx, float learning_step) = 0;

protected:
	Layer* _next_layer;
	Layer* _prev_layer;

	uint32_t _input_dim;
	uint32_t* _input_shape;
	uint32_t _output_dim;
	uint32_t* _output_shape;

	Tensor* _cached_input;

	void initInput(uint32_t input_dim, uint32_t* input_shape);
	void initOutput(uint32_t output_dim, uint32_t* output_shape);
};