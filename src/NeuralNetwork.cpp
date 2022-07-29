#include "NeuralNetwork.h"

extern double g_time;

NeuralNetwork::NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&), const Tensor(*cost_function_d)(const Tensor&, const Tensor&)) {
	_input_layer = &input_layer;
	_output_layer = &output_layer;
	_cost_function = cost_function;
	_cost_function_d = cost_function_d;
}

NeuralNetwork::NeuralNetwork(Layer& input_layer, Layer& output_layer, CostFun cost_fun) {
	_input_layer = &input_layer;
	_output_layer = &output_layer;
	switch (cost_fun) {
	case CostFun::BinaryCrossentropy:
		_cost_function = binary_crossentropy;
		_cost_function_d = binary_crossentropy_d;
		break;
	default:
		_cost_function = nullptr;
		_cost_function_d = nullptr;
		// exception
	}
}

float(*NeuralNetwork::getCostFun())(const Tensor&, const Tensor&) {
	return _cost_function;
}

const Tensor NeuralNetwork::predict(const Tensor& input) {
	Layer* layer;
	Tensor output;

	layer = _input_layer;
	output = layer->forwardPropagation(input);

	while (layer != _output_layer) {
		layer = layer->getNextLayer();
		output = layer->forwardPropagation(output);
	}

	return output;
}

FitHistory NeuralNetwork::fit(const Tensor& train_x, const Tensor& train_y, const Tensor& test_x, const Tensor& test_y, uint32_t batch_size, uint32_t epochs, float learning_step, uint8_t verbose) {
	FitHistory result;
	Layer* layer;
	uint32_t epoch{ 0 };
	uint32_t batch_start{ 0 };

	result.length = epochs;

	result.test_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result.test_cost) {
		// exception
	}
	result.train_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result.train_cost) {
		// exception
	}

	for (epoch = 0; epoch < epochs; ++epoch) {
		uint32_t* permutation = genPermutation(train_x.getShape()[0]);

		const Tensor train_x_shuffled = train_x.shuffle(permutation);
		const Tensor train_y_shuffled = train_y.shuffle(permutation);

		free(permutation);

		float train_cost{ 0.0f };
		float test_cost{ 0.0f };
		uint32_t batch_count{ 0 };


		double train_start = perf_counter_ns();
		for (batch_start = 0; batch_start + batch_size <= train_x.getShape()[0]; batch_start += batch_size) {
			initLayersCachedGradient();

			std::vector<std::vector<uint32_t>> slice_vector_x;
			for (uint32_t i{ 0 }; i < train_x_shuffled.getDim(); ++i) {
				slice_vector_x.push_back({});
			}
			slice_vector_x[0].push_back(batch_start);
			slice_vector_x[0].push_back(batch_start + batch_size);

			std::vector<std::vector<uint32_t>> slice_vector_y;
			for (uint32_t i{ 0 }; i < train_y_shuffled.getDim(); ++i) {
				slice_vector_y.push_back({});
			}
			slice_vector_y[0].push_back(batch_start);
			slice_vector_y[0].push_back(batch_start + batch_size);

			Tensor batch_x = train_x_shuffled[slice_vector_x];
			Tensor batch_y = train_y_shuffled[slice_vector_y];

			Tensor y_hat = predict(batch_x);

			float batch_cost = _cost_function(y_hat, batch_y);

			layer = _output_layer;

			Tensor dx = _cost_function_d(y_hat, batch_y);
			dx = layer->backwardPropagation(dx);

			while (layer != _input_layer) {
				layer = layer->getPrevLayer();
				std::vector<uint32_t> dx_new_shape = layer->getOutputShape();
				dx_new_shape.insert(dx_new_shape.begin(), dx.getShape()[0]);
				dx = dx.reshape(dx_new_shape);
				dx = layer->backwardPropagation(dx);
			}

			train_cost += batch_cost;
			++batch_count;

			uint32_t done = batch_start / batch_size + 1;
			uint32_t total = train_x.getShape()[0] / batch_size;
			
			if (verbose >= 1) {
				printf("\r%4d ", epoch + 1);
				print_progress(static_cast<float>(done) / total);
				printf(" ");
				print_time(TIME_DIFF_SEC(train_start, perf_counter_ns()) * (total - done) / done);
				printf(" train cost: %f", train_cost / batch_count);
				fflush(stdout);
			}

			updateLayersWeights(learning_step);

		}
		if (verbose >= 1) {
			printf("\r%4d ", epoch + 1);
			print_progress(1.0f);
			printf(" ");
			print_time(TIME_DIFF_SEC(train_start, perf_counter_ns()));
			printf(" train cost: %f", train_cost / batch_count);
			fflush(stdout);
		}

		result.train_cost[epoch] = train_cost / batch_count;

		batch_count = 0;
		uint32_t test_batch_size = batch_size < test_x.getShape()[0] ? batch_size : test_x.getShape()[0];
		for (batch_start = 0; batch_start + test_batch_size <= test_x.getShape()[0]; batch_start += test_batch_size) {
			std::vector<std::vector<uint32_t>> slice_vector_x;
			for (uint32_t i{ 0 }; i < test_x.getDim(); ++i) {
				slice_vector_x.push_back({});
			}
			slice_vector_x[0].push_back(batch_start);
			slice_vector_x[0].push_back(batch_start + batch_size);
			
			std::vector<std::vector<uint32_t>> slice_vector_y;
			for (uint32_t i{ 0 }; i < test_y.getDim(); ++i) {
				slice_vector_y.push_back({});
			}
			slice_vector_y[0].push_back(batch_start);
			slice_vector_y[0].push_back(batch_start + batch_size);

			Tensor batch_x = test_x[slice_vector_x];
			Tensor batch_y = test_y[slice_vector_y];

			float batch_cost = _cost_function(predict(batch_x), batch_y);

			test_cost += batch_cost;
			++batch_count;
		}
		if (verbose >= 1) {
			printf(" test cost: %f\n", test_cost / batch_count);
		}

		result.test_cost[epoch] = test_cost / batch_count;
	}

	return result;
}

void NeuralNetwork::summary() const {
	Layer *layer{ _input_layer };
	uint32_t total_params{ 0u };

	while (1) {
		layer->summary();
		total_params += layer->getParamsCount();
		if (layer == _output_layer) {
			break;
		}
		layer = layer->getNextLayer();
	}
	printf("total params: %d\n", total_params);
}

float NeuralNetwork::binary_crossentropy(const Tensor& y_hat, const Tensor& y) {
	Tensor result = y * (y_hat + 1e-9f).applyFunction(logf) + (-y + 1.0f) * (-y_hat + 1.0f + 1e-9f).applyFunction(logf);
	return result.sum() * (-1.0f / y.getSize());
}

const Tensor NeuralNetwork::binary_crossentropy_d(const Tensor& y_hat, const Tensor& y) {
	Tensor result = (y / (y_hat + 1e-9f)) - ((-y + 1.0f) / (-y_hat + 1.0f + 1e-9f));
	return -result;
}

void NeuralNetwork::updateLayersWeights(float learning_step) {
	Layer* layer;

	layer = _input_layer;
	layer->updateWeights(learning_step);

	while (layer != _output_layer) {
		layer = layer->getNextLayer();
		layer->updateWeights(learning_step);
	}
}

void NeuralNetwork::initLayersCachedGradient() {
	Layer* layer;

	layer = _input_layer;
	layer->initCachedGradient();

	while (layer != _output_layer) {
		layer = layer->getNextLayer();
		layer->initCachedGradient();
	}
}

void NeuralNetwork::print_progress(float percent) {
	uint32_t i{ 0 };
	bool arrow = true;

	printf("[");
	for (i = 0; i < 32; ++i) {
		if (i < 32 * percent) {
			printf("=");
		}
		else if (arrow) {
			printf(">");
			arrow = false;
		} else {
			printf(" ");
		}
	}
	printf("]");
}

void NeuralNetwork::print_time(double seconds_d) {
	uint32_t full_seconds = static_cast<uint32_t>(seconds_d);
	uint32_t hours = full_seconds / 3600;
	uint32_t minutes = (full_seconds % 3600) / 60;
	uint32_t seconds = full_seconds % 60;

	if (hours) {
		printf("%d:%02d:%02d", hours, minutes, seconds);
	} else if (minutes) {
		printf("%d:%02d", minutes, seconds);
	} else {
		printf("%.3f", seconds_d);
	}
}