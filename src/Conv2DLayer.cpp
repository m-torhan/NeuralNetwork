#include "Conv2DLayer.h"

Conv2DLayer::Conv2DLayer(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size) : Layer() {
	_input_shape = input_shape;
	if (2 == _input_shape.size()) {
		_input_shape.push_back(1);
	}
	else if (3 != _input_shape.size()) {
		// exception
	}
	_output_shape = _input_shape;
    _output_shape[_output_shape.size() - 1] = filters_count;
	_filters_count = filters_count;
	_filter_size = filter_size;
	initWeights(_input_shape, filters_count, filter_size);
}

Conv2DLayer::Conv2DLayer(Layer& prev_layer, uint32_t filters_count, uint32_t filter_size) : Layer() {
	_input_shape = prev_layer.getOutputShape();
	if (2 == _input_shape.size()) {
		_input_shape.push_back(1);
	}
	else if (3 != _input_shape.size()) {
		// exception
	}
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
	printf("Conv2D Layer        ");
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
	_cached_input = x;

	std::vector<uint32_t> x_next_pad_shape = _output_shape;
	x_next_pad_shape.insert(x_next_pad_shape.begin(), x.getShape()[0]);

	// padding
	x_next_pad_shape[1] += _filter_size - 1;
	x_next_pad_shape[2] += _filter_size - 1;

	Tensor x_next_pad(x_next_pad_shape);
	x_next_pad *= 0.0f;
	
	for (uint32_t i{ 0 }; i < _filter_size; ++i) {
		for (uint32_t j{ 0 }; j < _filter_size; ++j) {
			for (uint32_t f{ 0 }; f < _filters_count; ++f) {
				x_next_pad[{{},
							{ i, i + x.getShape()[1] },
							{ j, j + x.getShape()[2] },
							{ f }}] = const_cast<const Tensor&>(x_next_pad)[{{},
																			 { i, i + x.getShape()[1]},
																			 { j, j + x.getShape()[2]},
																			 { f }}] +
									  (x * const_cast<const Tensor&>(_weights)[{{ _filter_size - i - 1 },
									  											{ _filter_size - j - 1 },
																				{ 0, _weights.getShape()[2] },
																				{ f }}]).sum(3);
			}
		}
	}

	// remove padding
	Tensor x_next = const_cast<const Tensor&>(x_next_pad)[{ {},
													 		{_filter_size >> 1, x.getShape()[1] + (_filter_size >> 1)},
													 		{_filter_size >> 1, x.getShape()[2] + (_filter_size >> 1)},
													 		{} }];

	x_next += _biases;

	_cached_output = x_next;

	return x_next;
}

const Tensor Conv2DLayer::backwardPropagation(const Tensor& dx) {
	Tensor dx_prev(_cached_input);
	dx_prev *= 0.0f;

	_samples += _cached_input.getShape()[0];

	Tensor weights_d(_weights.getShape());
	weights_d *= 0.0f;
	Tensor biases_d(_biases.getShape());
	biases_d *= 0.0f;

	uint32_t padding_size = (_filter_size - 1) >> 1;

	for (uint32_t i{ 0 }; i < dx.getShape()[0]; ++i) {
		const Tensor cached_input_pad = const_cast<const Tensor&>(_cached_input)[{ {{ i }}, {}, {}, {} }].addPadding(
			{ 0, 1 }, { Both, Both }, { padding_size, padding_size });
		Tensor dx_prev_pad = const_cast<const Tensor&>(dx_prev)[{ {{ i }}, {}, {}, {} }].addPadding(
			{ 0, 1 }, { Both, Both }, { padding_size, padding_size });

		for (uint32_t h{ 0 }; h < dx.getShape()[1]; ++h) {
			for (uint32_t w{ 0 }; w < dx.getShape()[2]; ++w) {
				for (uint32_t c{ 0 }; c < dx.getShape()[3]; ++c) {
					Tensor sub_tensor = const_cast<const Tensor&>(dx_prev_pad)[{ { h, h + _filter_size }, { w, w + _filter_size }, {} }];
					sub_tensor += const_cast<const Tensor&>(_weights)[{ {}, {}, {}, {{c}} }]*dx[{ i, h, w, c }];
					dx_prev_pad[{ { h, h + _filter_size }, { w, w + _filter_size }, {} }] = sub_tensor;

					sub_tensor = const_cast<const Tensor&>(weights_d)[{ {}, {}, {}, {{c}} }];
					sub_tensor += cached_input_pad[{ { h, h + _filter_size }, { w, w + _filter_size }, {} }]*dx[{ i, h, w, c }];
					weights_d[{ {}, {}, {}, {{c}} }] = sub_tensor;

					biases_d[std::vector<uint32_t>({ c })] = const_cast<const Tensor&>(biases_d)[{ c }] + dx[{ i, h, w, c }];
				}
			}
		}

		dx_prev[{ {{i}}, {}, {}, {}}] = const_cast<const Tensor&>(dx_prev_pad)[{
			{ (_filter_size - 1) / 2, dx_prev_pad.getShape()[0] - (_filter_size - 1) / 2 },
			{ (_filter_size - 1) / 2, dx_prev_pad.getShape()[1] - (_filter_size - 1) / 2 },
			{} }];
	}

	_cached_weights_d += weights_d;
	_cached_biases_d += biases_d;

	return dx_prev;
}