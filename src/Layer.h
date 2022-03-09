#pragma once
#include "Tensor.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

class Layer {
public:
	~Layer();

	uint32_t getInputDim() const;
	std::vector<uint32_t> getInputShape() const;
	uint32_t getOutputDim() const;
	std::vector<uint32_t> getOutputShape() const;
	void setPrevLayer(Layer* layer);
	void setNextLayer(Layer* layer);
	Layer* getPrevLayer() const;
	Layer* getNextLayer() const;
	Tensor getCachedOutput() const;

	virtual const Tensor forwardPropagation(const Tensor& x) = 0;
	virtual const Tensor backwardPropagation(const Tensor& dx) = 0;
	virtual void updateWeights(float learning_step) = 0;
	virtual void initCachedGradient() = 0;

protected:
	Layer* _next_layer;
	Layer* _prev_layer;

	std::vector<uint32_t> _input_shape;
	std::vector<uint32_t> _output_shape;

	Tensor _cached_input;
	Tensor _cached_output;
};