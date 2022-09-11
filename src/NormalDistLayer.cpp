#include "NormalDistLayer.h"

NormalDistLayer::NormalDistLayer(std::vector<uint32_t> input_shape) {
    if (input_shape.size() != 1) {
        // exception
    }
    _input_shape = input_shape;
    input_shape[input_shape.size() - 1] = 1;
    _output_shape = input_shape;
}

NormalDistLayer::NormalDistLayer(Layer& prev_layer) {
    std::vector<uint32_t> prev_output_shape = prev_layer.getOutputShape();
    if (prev_output_shape.size() != 1) {
        // exception
    }
    _input_shape = prev_output_shape;
    prev_output_shape[prev_output_shape.size() - 1] = 1;
    _output_shape = prev_output_shape;

	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

const Tensor NormalDistLayer::forwardPropagation(const Tensor& x) {
    _cached_input = x;

	std::vector<uint32_t> random_tensor_shape = x.getShape();
    random_tensor_shape.pop_back();

    Tensor random_tensor = Tensor::RandomNormal(random_tensor_shape);

    std::vector<std::vector<uint32_t>> mu_slice;
    std::vector<std::vector<uint32_t>> var_slice;

    for (uint32_t i{ 0 }; i < random_tensor_shape.size(); ++i) {
        mu_slice.push_back({0, random_tensor_shape[i]});
        var_slice.push_back({0, random_tensor_shape[i]});
    }

    mu_slice.push_back({ 0 });
    var_slice.push_back({ 1 });

    Tensor x_next = (x[mu_slice] + random_tensor*x[var_slice]).reshape({ x.getShape()[0], 1 });

    _cached_random_tensor = random_tensor;
    _cached_output = x_next;
    return x_next;
}

const Tensor NormalDistLayer::backwardPropagation(const Tensor& dx) {
    std::vector<uint32_t> dx_prev_shape = dx.getShape();
    dx_prev_shape.pop_back();
    dx_prev_shape.push_back(2);

    Tensor dx_prev(dx_prev_shape);

    std::vector<std::vector<uint32_t>> mu_slice;
    std::vector<std::vector<uint32_t>> var_slice;

    for (uint32_t i{ 0 }; i < dx.getShape().size() - 1; ++i) {
        mu_slice.push_back({0, dx.getShape()[i]});
        var_slice.push_back({0, dx.getShape()[i]});
    }

    mu_slice.push_back({ 0, 1 });
    var_slice.push_back({ 1, 2 });

    dx_prev[mu_slice] = dx;
    dx_prev[var_slice] = const_cast<const Tensor&>(_cached_input)[{var_slice}]*dx;

    return dx_prev;
}

void NormalDistLayer::updateWeights(float learning_step) {}

void NormalDistLayer::initCachedGradient() {}

void NormalDistLayer::summary() const {
	printf("NormDistLayer Layer ");
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

uint32_t NormalDistLayer::getParamsCount() const {
    return 0;
}
