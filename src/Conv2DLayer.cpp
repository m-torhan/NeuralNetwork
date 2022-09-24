#include "Conv2DLayer.h"

Conv2DLayer::Conv2DLayer(std::vector<uint32_t> input_shape, uint32_t filters_count, uint32_t filter_size) : Layer() {
	_input_shape = input_shape;
	if (2 == _input_shape.size()) {
		_input_shape.push_back(1);
	}
	else if (3 != _input_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Invalid input shape. Dim should be 3, but is %d.",
			__FILE__, __LINE__, _input_shape.size()));
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
		throw std::invalid_argument(format_string("%s %d : Invalid input shape. Dim should be 3, but is %d.",
			__FILE__, __LINE__, _input_shape.size()));
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

	_weights.applyFunction([](float value) {return randNormalDistribution(); });
	_weights /=  filter_size * filter_size;
    //_weights.applyFunction([](float value) {return randUniform(-1.0f, 1.0f) * sqrtf(6.0f); });

	_biases = Tensor({ filters_count });
	_biases *= 0.0f;

    //_biases.applyFunction([](float value) {return randUniform(-1.0f, 1.0f) * sqrtf(6.0f); });
	
	_cached_weights_d_velocity = Tensor(_weights.getShape());
	_cached_biases_d_velocity = Tensor(_biases.getShape());
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

void Conv2DLayer::updateWeights(float learning_step, float momentum) {
	_cached_weights_d_velocity = (_cached_weights_d * learning_step / _samples) + momentum * _cached_weights_d_velocity;
	_cached_biases_d_velocity = (_cached_biases_d * learning_step / _samples) + momentum * _cached_biases_d_velocity;

	_weights -= _cached_weights_d_velocity;
	_biases -= _cached_biases_d_velocity;
}

const Tensor Conv2DLayer::forwardPropagation(const Tensor& x, bool inference) {

	const Tensor x_pad = x.addPadding({ 1, 2 }, { Both, Both }, { (_filter_size - 1) >> 1, (_filter_size - 1) >> 1 });

	uint32_t batch_size = x.getShape()[0]; 	// b
	uint32_t height = x.getShape()[1];		// h
	uint32_t width = x.getShape()[2];		// w
	uint32_t channels = x.getShape()[3];	// c

	Tensor rects = Tensor({ batch_size * height * width, _filter_size * _filter_size * channels });

	for (uint32_t i{ 0 }; i < batch_size; ++i) {
		for (uint32_t j{ 0 }; j < height; ++j) {
			for (uint32_t k{ 0 }; k < width; ++k) {
				rects[{ { i * width * height + j * width + k }, { 0, _filter_size * _filter_size * channels }}] =
					x_pad[{{ i }, { j, j + _filter_size }, { k, k + _filter_size }, { 0, channels } }].reshape({ _filter_size * _filter_size * channels });
			}
		}
	}
	
	// [bhw, ssc] * [ssc, f] = [bhw, f] -> [b, h, w, f]
	Tensor x_next = std::move(rects.dotProductTranspose(_weights.reshape({ _filter_size * _filter_size * channels, _filters_count }).transpose()).reshape({ batch_size, height, width, _filters_count }));

	x_next += _biases;

	if (!inference)
	{
		_cached_input = x;
		_cached_output = x_next;
		_cached_rects = rects;
	}

	return x_next;
}

const Tensor Conv2DLayer::backwardPropagation(const Tensor& dx) {
	_samples += _cached_input.getShape()[0];

	uint32_t batch_size = _cached_input.getShape()[0];	// b
	uint32_t height = _cached_input.getShape()[1];		// h
	uint32_t width = _cached_input.getShape()[2];		// w
	uint32_t channels = _cached_input.getShape()[3];	// c

	const Tensor x_pad = _cached_input.addPadding({ 1, 2 }, { Both, Both }, { (_filter_size - 1) >> 1, (_filter_size - 1) >> 1 });
	const Tensor dx_pad = dx.addPadding({ 1, 2 }, { Both, Both }, { (_filter_size - 1) >> 1, (_filter_size - 1) >> 1 });

	Tensor x_rects = _cached_rects;
	Tensor dx_rects = Tensor({ batch_size * height * width, _filter_size * _filter_size * _filters_count });
	
	for (uint32_t i{ 0 }; i < batch_size; ++i) {
		for (uint32_t j{ 0 }; j < height; ++j) {
			for (uint32_t k{ 0 }; k < width; ++k) {
				// for (uint32_t i_x{ 0 }; i_x < _filter_size; ++i_x) {
				// 	for (uint32_t i_y{ 0 }; i_y < _filter_size; ++i_y) {
				// 		for (uint32_t i_c{ 0 }; i_c < _filters_count; ++i_c) {
				// 			dx_rects[{ { i * width * height + j * width + k }, { i_x * _filter_size * _filters_count + i_y * _filters_count + i_c }}] =
				// 				dx_pad[{{ i }, { j + (_filter_size - 1 - i_x) }, { k + (_filter_size - 1 - i_y) }, { i_c } }];
				// 		}
				// 	}
				// }
				dx_rects[{ { i * width * height + j * width + k }, { 0, _filter_size * _filter_size * _filters_count }}] =
					dx_pad[{{ i }, { j, j + _filter_size }, { k, k + _filter_size }, { 0, _filters_count } }].reshape({ _filter_size * _filter_size * _filters_count });
			}
		}
	}

	// [ssc, bhw] * [bhw, f] = [ssc, f] -> [s, s, c, f]
	Tensor weights_d = std::move(x_rects.transpose().dotProduct(dx.reshape({ batch_size * width * height, _filters_count })).reshape(_weights.getShape()));
	Tensor biases_d({ _filters_count });

	for (uint32_t i{ 0 }; i < _filters_count; ++i) {
		biases_d[{ i }] = dx[{ { 0, batch_size }, { 0, height }, { 0, width }, { i }}].sum();
	}

	_cached_weights_d += weights_d;
	_cached_biases_d += biases_d;

	// [s, s, c, f] -> [s, s, f, c]
	Tensor filters_transformed({ _filter_size, _filter_size, _filters_count, channels });

	for (uint32_t i{ 0 }; i < _filter_size; ++i) {
		for (uint32_t j{ 0 }; j < _filter_size; ++j) {
			filters_transformed[{ { i }, { j }, { 0, _filters_count}, { 0, channels }}] =
				const_cast<const Tensor&>(_weights)[{ { i }, { j }, { 0, channels }, { 0, _filters_count} }].transpose();
		}
	}

	// [s, s, f, c] -> [ssf, c] -> [c, ssf]
	filters_transformed = std::move(filters_transformed.reshape({ _filter_size * _filter_size * _filters_count, channels }).transpose());

	// [bhw, ssf] * [ssf, c] = [bhw, c] -> [b, h, w, c]
	Tensor dx_prev = std::move(dx_rects.dotProductTranspose(filters_transformed).reshape({ batch_size, height, width, channels }));

	return dx_prev;
}