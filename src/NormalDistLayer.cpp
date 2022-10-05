#include "NormalDistLayer.h"

NormalDistLayer::NormalDistLayer(std::vector<uint32_t> input_shape) {
    if (input_shape.size() != 2) {
		throw std::invalid_argument(format_string("%s %d : Invalid input shape. Dim should be 2, but is %d.",
            __FILE__, __LINE__, _input_shape.size()));
    }
    _input_shape = input_shape;
    input_shape[input_shape.size() - 1] = 2;
    _output_shape = input_shape;
    input_shape[input_shape.size() - 1] = 1;
}

NormalDistLayer::NormalDistLayer(Layer& prev_layer) {
    std::vector<uint32_t> prev_output_shape = prev_layer.getOutputShape();
    if (prev_output_shape.size() != 2) {
		throw std::invalid_argument(format_string("%s %d : Invalid input shape. Dim should be 2, but is %d.",
            __FILE__, __LINE__, prev_output_shape.size()));
    }
    _input_shape = prev_output_shape;
    prev_output_shape[prev_output_shape.size() - 1] = 1;
    _output_shape = prev_output_shape;

	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

const Tensor NormalDistLayer::forwardPropagation(const Tensor& x, bool inference) {

	std::vector<uint32_t> random_tensor_shape = x.getShape();
    random_tensor_shape.pop_back(); // [...]

    Tensor random_tensor = Tensor::RandomNormal(random_tensor_shape);

    std::vector<std::vector<uint32_t>> mu_slice;
    std::vector<std::vector<uint32_t>> var_slice;

    for (uint32_t i{ 0 }; i < random_tensor_shape.size(); ++i) {
        mu_slice.push_back({0, random_tensor_shape[i]});
        var_slice.push_back({0, random_tensor_shape[i]});
    }

    mu_slice.push_back({ 0 }); //  [..., 0]
    var_slice.push_back({ 1 }); // [..., 1]

    std::vector<uint32_t> x_next_shape = x.getShape();
    // [..., 1]
    x_next_shape[x_next_shape.size() - 1] = 1;

    Tensor x_next = (x[mu_slice] + random_tensor*((0.5f * x[var_slice]).applyFunction(expf))).reshape(x_next_shape);

    if (!inference)
    {
        _cached_input = x;
        _cached_random_tensor = random_tensor;
        _cached_output = x_next;
    }
    return x_next;
}

const Tensor NormalDistLayer::backwardPropagation(const Tensor& dx) {
    std::vector<uint32_t> dx_prev_shape = dx.getShape();
    dx_prev_shape[dx_prev_shape.size() - 1] = 2;

    Tensor dx_prev(dx_prev_shape);

    std::vector<std::vector<uint32_t>> mu_slice;
    std::vector<std::vector<uint32_t>> var_slice;

    for (uint32_t i{ 0 }; i < dx.getShape().size() - 1; ++i) {
        mu_slice.push_back({0, dx.getShape()[i]});
        var_slice.push_back({0, dx.getShape()[i]});
    }

    mu_slice.push_back({ 0, 1 });
    var_slice.push_back({ 1, 2 });

    const Tensor input_var = const_cast<const Tensor&>(_cached_input)[{var_slice}];
    const Tensor input_mu = const_cast<const Tensor&>(_cached_input)[{mu_slice}];

    constexpr float KL_coef = .0001f;

    dx_prev[mu_slice] = dx - KL_coef*input_mu;
    dx_prev[var_slice] = input_var*dx + KL_coef*0.5f*(1 - input_var.applyFunction(expf));

    return dx_prev;
}

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
	printf(")\n");
}

uint32_t NormalDistLayer::getParamsCount() const {
    return 0;
}
