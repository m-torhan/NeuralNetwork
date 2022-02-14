#include "DenseLayer.h"

#include <cstdlib>
#include <cstring>

DenseLayer::DenseLayer(uint32_t input_dim, uint32_t* input_shape, uint32_t neurons_count) {

}

Tensor* DenseLayer::forwardPropagation(const Tensor& tensor) {
	Tensor* result = new Tensor(tensor);

	return result;
}

Tensor* DenseLayer::backwardPropagation(const Tensor& tensor) {
	Tensor* result = new Tensor(tensor);

	return result;
}