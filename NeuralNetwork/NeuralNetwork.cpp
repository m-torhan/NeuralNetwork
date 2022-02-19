#include "NeuralNetwork.h"

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>

NeuralNetwork::NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&), Tensor*(*cost_function_d)(const Tensor&, const Tensor&)) {
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

Tensor* NeuralNetwork::predict(Tensor* input) {
	Layer* layer;
	Tensor* output;

	layer = _input_layer;
	output = layer->forwardPropagation(*input);

	while (layer != _output_layer) {
		layer = layer->getNextLayer();
		output = layer->forwardPropagation(*output);
	}

	return output;
}

FitHistory* NeuralNetwork::fit(Tensor* train_x, Tensor* train_y, Tensor* test_x, Tensor* test_y, uint32_t batch_size, uint32_t epochs, float learning_step) {
	FitHistory* result;
	Layer* layer;
	Tensor* dx;
	Tensor* batch_x;
	Tensor* batch_y;
	Tensor* y_hat;
	uint32_t batch_count = 0;
	uint32_t epoch = 0;
	uint32_t batch_start = 0;
	uint32_t* permutation;
	float train_cost;
	float test_cost;
	float batch_cost;
	double train_start;
	uint32_t done;
	uint32_t total;

	result = (FitHistory*)malloc(sizeof(FitHistory));
	if (!result) {
		// exception
	}
	result->test_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result->test_cost) {
		// exception
	}
	result->train_cost = (float*)malloc(sizeof(float) * epochs);
	if (!result->train_cost) {
		// exception
	}

	for (epoch = 0; epoch < epochs; ++epoch) {
		permutation = genPermutation(train_x->getShape()[0]);

		train_x = train_x->shuffle(permutation);
		train_y = train_y->shuffle(permutation);

		train_cost = 0;
		test_cost = 0;

		initLayersCachedGradient();

		train_start = perf_counter_ns();

		for (batch_start = 0; batch_start + batch_size <= train_x->getShape()[0]; batch_start += batch_size) {
			batch_x = train_x->slice(0, batch_start, batch_start + batch_size);
			batch_y = train_y->slice(0, batch_start, batch_start + batch_size);

			y_hat = predict(batch_x);

			batch_cost = _cost_function(*y_hat, *batch_y);

			layer = _output_layer;

			dx = _cost_function_d(*y_hat, *batch_y);
			dx = layer->backwardPropagation(*dx, learning_step);

			while (layer != _input_layer) {
				layer = layer->getPrevLayer();
				dx = layer->backwardPropagation(*dx, learning_step);
			}

			train_cost += batch_cost;
			++batch_count;

			done = batch_start / batch_size + 1;
			total = train_x->getShape()[0] / batch_size;

			printf("\r %d ", epoch);
			print_progress(static_cast<float>(done) / total);
			printf(" ");
			print_time(TIME_DIFF_SEC(train_start, perf_counter_ns()) * (total - done) / done);
			printf(" train cost: %f", train_cost / batch_count);
		}

		updateLayersWeights();

		batch_count = 0;
		for (batch_start = 0; batch_start + batch_size <= test_x->getShape()[0]; batch_start += batch_size) {
			batch_x = test_x->slice(0, batch_start, batch_start + batch_size);
			batch_y = test_y->slice(0, batch_start, batch_start + batch_size);

			batch_cost = _cost_function(*predict(batch_x), *batch_y);

			test_cost += batch_cost;
			++batch_count;
		}
		printf(" test cost: %f\n", test_cost / batch_count);
	}

	return result;
}

void NeuralNetwork::updateLayersWeights() {
	Layer* layer;

	layer = _input_layer;
	layer->updateWeights();

	while (layer != _output_layer) {
		layer = layer->getNextLayer();
		layer->updateWeights();
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

double NeuralNetwork::perf_counter_ns() {
	return (long double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

float NeuralNetwork::binary_crossentropy(const Tensor& y_hat, const Tensor& y) {
	Tensor* result;
	result = *y.dotProduct(*(y_hat + 1e-9f)->applyFunction(logf)) + *(*(*-y + 1.0f)).dotProduct(*(*(*-y_hat + 1.0f) + 1e-9f)->applyFunction(logf));
	return result->sum() * (-1.0f / y.getShape()[0]);
}

Tensor* NeuralNetwork::binary_crossentropy_d(const Tensor& y_hat, const Tensor& y) {
	Tensor* result;
	result = *(y / *(y_hat + 1e-9f)) - *(*(*-y + 1.0f) / *(*(*-y_hat + 1.0f) + 1e-9f));
	return *result * (-1.0f / y.getShape()[0]);
}