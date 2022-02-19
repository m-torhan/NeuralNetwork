#pragma once

#include "Layer.h"
#include "Utils.h"

#define TIME_DIFF_SEC(t_start, t_end) (float(t_end - t_start) / (CLOCKS_PER_SEC * 1000000))

typedef struct FitHistory {
	float* train_cost;
	float* test_cost;
};

enum class CostFun {
	BinaryCrossentropy
};

class NeuralNetwork {
public:
	NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&), Tensor*(*_cost_function)(const Tensor&, const Tensor&));
	NeuralNetwork(Layer& input_layer, Layer& output_layer, CostFun cost_fun);

	float(*getCostFun())(const Tensor&, const Tensor&);
	Tensor* predict(Tensor* input);
	FitHistory* fit(Tensor* train_x, Tensor* train_y, Tensor* test_x, Tensor* test_y, uint32_t batch_size, uint32_t epochs, float learning_step);

private:
	Layer* _input_layer;
	Layer* _output_layer;
	float(*_cost_function)(const Tensor& y_hat, const Tensor& y);
	Tensor*(*_cost_function_d)(const Tensor& y_hat, const Tensor& y);

	void updateLayersWeights();
	void initLayersCachedGradient();

	static void print_progress(float percent);
	static void print_time(double seconds);
	static double perf_counter_ns();

	static float binary_crossentropy(const Tensor& y_hat, const Tensor& y);
	static Tensor* binary_crossentropy_d(const Tensor& y_hat, const Tensor& y);
};