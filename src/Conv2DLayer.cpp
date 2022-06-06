#include "Conv2DLayer.h"

Conv2DLayer::Conv2DLayer(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size) : Layer() {
	_input_shape = input_shape;
	_output_shape = input_shape;
    _output_shape[_output_shape.size() - 1] = filters_count;
	_filters_count = filters_count;
	_filter_size = filter_size;
	initWeights(_input_shape, filters_count, filter_size);
}

Conv2DLayer::Conv2DLayer(Layer& prev_layer, uint32_t filters_count, uint32_t filter_size) : Layer() {
	_input_shape = prev_layer.getOutputShape();
	_output_shape = _input_shape;
    _output_shape[_output_shape.size() - 1] = filters_count;
	_filters_count = filters_count;
	_filter_size = filter_size;
	initWeights(_input_shape, filters_count, filter_size);
	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);
}

void Conv2DLayer::setWeights(std::vector<float> weights) {
	_weights.setValues(weights);
}

void Conv2DLayer::setBiases(std::vector<float> biases) {
	_biases.setValues(biases);
}

void Conv2DLayer::initWeights(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size) {
	_filters_count = filters_count;

	_weights = Tensor({ filter_size, filter_size, input_shape[2], filters_count });

    _weights.applyFunction([](float value) {return randUniform(-1.0f, 1.0f) * sqrtf(6.0f); });

	_biases = Tensor({ filters_count });

    _biases.applyFunction([](float value) {return randUniform(-1.0f, 1.0f) * sqrtf(6.0f); });
}

void Conv2DLayer::initCachedGradient() {
	_cached_weights_d = Tensor(_weights);
	_cached_biases_d = Tensor(_biases);
	_cached_weights_d *= 0.0f;
	_cached_biases_d *= 0.0f;
	_samples = 0;
}

void Conv2DLayer::summary() const {
	printf("Conv2D Layer\n");
	printf("  in shape:  (*");
	for (uint32_t i{ 0u }; i < _input_shape.size(); ++i) {
		printf(", %d", _input_shape[i]);
	}
	printf(")  ");
	printf("  out shape: (*");
	for (uint32_t i { 0u }; i < _output_shape.size(); ++i) {
		printf(", %d", _output_shape[i]);
	}
	printf(")  total params: %d\n", _weights.getSize() + _biases.getSize());
}

uint32_t Conv2DLayer::getParamsCount() const {
	return _weights.getSize() + _biases.getSize();
}

void Conv2DLayer::updateWeights(float learning_step) {
	_weights -= _cached_weights_d * learning_step / _samples;
	_biases -= _cached_biases_d * learning_step / _samples;
}

const Tensor Conv2DLayer::forwardPropagation(const Tensor& x) {
	_cached_input = Tensor(x);

	std::vector<uint32_t> x_next_shape = _output_shape;
	x_next_shape.insert(x_next_shape.begin(), x.getShape()[0]);

	Tensor x_next = Tensor(x_next_shape);

	for (uint32_t i{ 0 }; i < x.getShape()[0]; ++i) {
		Tensor sub_tensor_x = x.getSubTensor({ i, WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS });
    	Tensor sub_tensor_x_next = sub_tensor_x.addPadding({ 0, 1 }, { Both, Both }, { (_filter_size - 1) / 2, (_filter_size - 1) / 2 }).Conv2D(_weights) + _biases;
		x_next.setValuesOfSubTensor({i, WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS }, sub_tensor_x_next);
	}

	_cached_output = Tensor(x_next);

	return x_next;
}

const Tensor Conv2DLayer::backwardPropagation(const Tensor& dx) {
	Tensor dx_prev = Tensor(_cached_input);
	dx_prev *= 0.0f;

	_samples += _cached_input.getShape()[0];

	Tensor weights_d = Tensor(_weights.getShape());
	weights_d *= 0.0f;
	Tensor biases_d = Tensor(_biases.getShape());
	biases_d *= 0.0f;

	uint32_t padding_size = (_filter_size - 1) / 2;

	for (uint32_t i{ 0 }; i < dx.getShape()[0]; ++i) {
		Tensor cached_input_pad = _cached_input.getSubTensor({ i, WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS })
											   .addPadding({ 0, 1 }, { Both, Both }, { padding_size, padding_size });
		Tensor dx_prev_pad = dx_prev.getSubTensor({i, WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS})
									.addPadding({ 0, 1 }, { Both, Both }, { padding_size, padding_size });

		for (uint32_t h{ 0 }; h < dx.getShape()[1]; ++h) {
			for (uint32_t w{ 0 }; w < dx.getShape()[2]; ++w) {
				for (uint32_t c{ 0 }; c < dx.getShape()[3]; ++c) {
					Tensor sub_tensor = dx_prev_pad.getSubTensor({ { h, h + _filter_size }, { w, w + _filter_size }, {} });
					sub_tensor += _weights.getSubTensor({ WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS, c }) * dx.getValue({ i, h, w, c });
					dx_prev_pad.setValuesOfSubTensor({ { h, h + _filter_size }, { w, w + _filter_size }, {} }, sub_tensor);

					sub_tensor = weights_d.getSubTensor({ WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS, c });
					sub_tensor += cached_input_pad.getSubTensor({ { h, h + _filter_size }, { w, w + _filter_size }, {} }) * dx.getValue({ i, h, w, c });
					weights_d.setValuesOfSubTensor({ WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS, c }, sub_tensor);

					float bias_c = biases_d.getValue({ c });
					bias_c += dx.getValue({ i, h, w, c });
					biases_d.setValue(bias_c, { c });
				}
			}
		}

		dx_prev.setValuesOfSubTensor({ i, WHOLE_AXIS, WHOLE_AXIS, WHOLE_AXIS },
									 dx_prev_pad.getSubTensor({ { (_filter_size - 1) / 2, dx_prev_pad.getShape()[0] - (_filter_size - 1) / 2 },
									 							{ (_filter_size - 1) / 2, dx_prev_pad.getShape()[1] - (_filter_size - 1) / 2 },
																{} }));
	}

	_cached_weights_d += weights_d;
	_cached_biases_d += biases_d;

	return dx_prev;
}