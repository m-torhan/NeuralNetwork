#pragma once

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <thread>

#include "Layer.h"
#include "Utils.h"

#define TIME_DIFF_SEC(t_start, t_end) (float(t_end - t_start) / (CLOCKS_PER_SEC * 1000LL))

struct FitHistory {
	uint32_t length;
	float* train_cost;
	float* test_cost;
};

enum class CostFun {
	BinaryCrossentropy,
	CategoricalCrossentropy,
	MSE
};

class NeuralNetwork {
public:
	NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&), const Tensor(*cost_function_d)(const Tensor&, const Tensor&));
	NeuralNetwork(Layer& input_layer, Layer& output_layer, CostFun cost_fun);

	float(*getCostFun())(const Tensor&, const Tensor&);
	const Tensor predict(const Tensor& input, bool inference=true);
	FitHistory fit(const Tensor& train_x, const Tensor& train_y, const Tensor& test_x, const Tensor& test_y, uint32_t batch_size, uint32_t epochs, float learning_step, float momentum, uint8_t verbose=1u);

	void summary() const;

	static float binary_crossentropy(const Tensor& y_hat, const Tensor& y);
	static const Tensor binary_crossentropy_d(const Tensor& y_hat, const Tensor& y);
	static float categorical_crossentropy(const Tensor& y_hat, const Tensor& y);
	static const Tensor categorical_crossentropy_d(const Tensor& y_hat, const Tensor& y);
	static float mse(const Tensor& y_hat, const Tensor& y);
	static const Tensor mse_d(const Tensor& y_hat, const Tensor& y);

private:
	Layer* _input_layer;
	Layer* _output_layer;
	float(*_cost_function)(const Tensor& y_hat, const Tensor& y);
	const Tensor (*_cost_function_d)(const Tensor& y_hat, const Tensor& y);

	void updateLayersWeights(float learning_step, float momentum);
	void initLayersCachedGradient();

	static void print_progress(float percent);
	static void print_time(double seconds);
};