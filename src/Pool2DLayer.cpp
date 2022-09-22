#include "Pool2DLayer.h"

Pool2DLayer::Pool2DLayer(std::vector<uint32_t> input_shape, int32_t pool_size, PoolMode pool_mode) {
	_input_shape = input_shape;
	if (2 == _input_shape.size()) {
		_input_shape.push_back(1);
	}
	else if (3 != _input_shape.size()) {
		throw std::invalid_argument(format_string("Invalid input shape. Dim should be 3, but is {}.", _input_shape.size()));
	}

    if ((_input_shape[0] % pool_size) != 0) {
		throw std::invalid_argument(format_string("Invalid input shape. First dimension should be divisible by pool_size, but shape[0]={}, and pool_size={}", _input_shape[0], pool_size));
    }
    if ((_input_shape[1] % pool_size) != 0) {
		throw std::invalid_argument(format_string("Invalid input shape. Second dimension should be divisible by pool_size, but shape[1]={}, and pool_size={}", _input_shape[1], pool_size));
    }

	_output_shape = _input_shape;

    _output_shape[_output_shape.size() - 2] /= pool_size;
    _output_shape[_output_shape.size() - 3] /= pool_size;

    _pool_size = pool_size;
    switch (pool_mode) {
        case PoolMode::Max:
            _pool_function = pool_max;
            _pool_function_d = pool_max_d;
            break;
        case PoolMode::Average:
            _pool_function = pool_mean;
            _pool_function_d = pool_mean_d;
            break;
    }
}

Pool2DLayer::Pool2DLayer(Layer& prev_layer, int32_t pool_size, PoolMode pool_mode) {
	_input_shape = prev_layer.getOutputShape();
	if (2 == _input_shape.size()) {
		_input_shape.push_back(1);
	}
	else if (3 != _input_shape.size()) {
		throw std::invalid_argument(format_string("Invalid input shape. Dim should be 3, but is {}.", _input_shape.size()));
	}

    if ((_input_shape[_input_shape.size() - 2] % pool_size) != 0) {
		throw std::invalid_argument(format_string("Invalid input shape. First dimension should be divisible by pool_size, but shape[0]={}, and pool_size={}", _input_shape[0], pool_size));
    }
    if ((_input_shape[_input_shape.size() - 3] % pool_size) != 0) {
		throw std::invalid_argument(format_string("Invalid input shape. Second dimension should be divisible by pool_size, but shape[1]={}, and pool_size={}", _input_shape[1], pool_size));
    }

	_output_shape = _input_shape;

    _output_shape[_output_shape.size() - 2] /= pool_size;
    _output_shape[_output_shape.size() - 3] /= pool_size;

	this->setPrevLayer(&prev_layer);
	prev_layer.setNextLayer(this);

    _pool_size = pool_size;
    switch (pool_mode) {
        case PoolMode::Max:
            _pool_function = pool_max;
            _pool_function_d = pool_max_d;
            break;
        case PoolMode::Average:
            _pool_function = pool_mean;
            _pool_function_d = pool_mean_d;
            break;
    }
}

const Tensor Pool2DLayer::forwardPropagation(const Tensor& x, bool inference) {
    std::vector<uint32_t> x_shape = x.getShape();
    std::vector<uint32_t> new_shape = {
        1,
        x_shape[x_shape.size() - 3],
        x_shape[x_shape.size() - 2],
        x_shape[x_shape.size() - 1] };
    
    for (uint32_t i = 0; i < x_shape.size() - 3; ++i) {
        new_shape[0] *= x_shape[i];
    }

    const Tensor reshaped_x = x.reshape(new_shape);

    std::vector<uint32_t> reshaped_result_shape = new_shape;
    reshaped_result_shape[1] /= _pool_size;
    reshaped_result_shape[2] /= _pool_size;
    Tensor reshaped_result = Tensor(reshaped_result_shape);

    for (uint32_t i = 0; i < reshaped_result_shape[0]; ++i) {
        for (uint32_t x = 0; x < reshaped_result_shape[1]; ++x) {
            for (uint32_t y = 0; y < reshaped_result_shape[2]; ++y) {
                for (uint32_t c = 0; c < reshaped_result_shape[3]; ++c) {
                    reshaped_result[{ i, x, y, c }] =
                        _pool_function(reshaped_x[
                            { { i },
                            { _pool_size*x, _pool_size*(x + 1) },
                            { _pool_size*y, _pool_size*(y + 1) },
                            { c }}]);
                }
            }
        }
    }
    std::vector<uint32_t> result_shape = x_shape;
    result_shape[result_shape.size() - 3] = x_shape[x_shape.size() - 3]/_pool_size;
    result_shape[result_shape.size() - 2] = x_shape[x_shape.size() - 2]/_pool_size;

    Tensor result = reshaped_result.reshape(result_shape);

    if (!inference)
    {
        _cached_input = x;
        _cached_output = result;
    }
    return result;
}

const Tensor Pool2DLayer::backwardPropagation(const Tensor& dx) {
    std::vector<uint32_t> x_shape = _cached_input.getShape();
    std::vector<uint32_t> new_shape = {
        1,
        x_shape[x_shape.size() - 3],
        x_shape[x_shape.size() - 2],
        x_shape[x_shape.size() - 1] };
    
    for (uint32_t i = 0; i < x_shape.size() - 3; ++i) {
        new_shape[0] *= x_shape[i];
    }

    std::vector<uint32_t> reshaped_result_shape = new_shape;
    reshaped_result_shape[1] /= _pool_size;
    reshaped_result_shape[2] /= _pool_size;
    Tensor reshaped_result = Tensor(new_shape);

    const Tensor reshaped_cached_input = std::move(_cached_input.reshape(new_shape));
    const Tensor reshaped_dx = std::move(dx.reshape(reshaped_result_shape));
    
    for (uint32_t i = 0; i < reshaped_result_shape[0]; ++i) {
        for (uint32_t x = 0; x < reshaped_result_shape[1]; ++x) {
            for (uint32_t y = 0; y < reshaped_result_shape[2]; ++y) {
                for (uint32_t c = 0; c < reshaped_result_shape[3]; ++c) {
                    reshaped_result[
                        { { i },
                          { _pool_size*x, _pool_size*(x + 1) },
                          { _pool_size*y, _pool_size*(y + 1) },
                          { c } }] =
                        _pool_function_d(
                            reshaped_cached_input[
                                { { i },
                                  { _pool_size*x, _pool_size*(x + 1) },
                                  { _pool_size*y, _pool_size*(y + 1) },
                                  { c } }],
                            reshaped_dx[{ i, x, y, c }]);
                }
            }
        }
    }
    
    Tensor result = std::move(reshaped_result.reshape(x_shape));

    return result;
}

void Pool2DLayer::summary() const {
	printf("Pool2D Layer        ");
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

uint32_t Pool2DLayer::getParamsCount() const {
    return 0;
}

float Pool2DLayer::pool_max(const Tensor& x) {
    return x.max();
}

const Tensor Pool2DLayer::pool_max_d(const Tensor& x, float dx) {
    float max = x.max();
    Tensor result = x;

    std::vector<float> x_data = x.getData();

    std::transform(x_data.begin(), x_data.end(), x_data.begin(),
        [max, dx] (float x) {return x == max ? dx : 0.0f;});

    float c = 0.0f;
    for (auto v : x_data) {
        if (v != 0) {
            c += 1;
        }
    }

    if (c > 0) {
        std::transform(x_data.begin(), x_data.end(), x_data.begin(),
            [c] (float x) {return x/c;});
    }

    result.setValues(x_data);

    return result;
}

float Pool2DLayer::pool_mean(const Tensor& x) {
    return x.mean();
}

const Tensor Pool2DLayer::pool_mean_d(const Tensor& x, float dx) {
    return (x/x.getSize())*dx;
}