#pragma once

#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <cmath>
#include <thread>

#include "Layer.h"
#include "Utils.h"

#define TIME_DIFF_SEC(t_start, t_end) (float(t_end - t_start) / (CLOCKS_PER_SEC * 1000LL))

/**
 * @brief Struct that contains cost functions values during the NN training.
 * 
 */
struct FitHistory {
	/**
	 * Length of the history.
	 */
	uint32_t length;
	/**
	 * Cost function values for train data.
	 */
	float* train_cost;
	/**
	 * Cost function values for test data.
	 */
	float* test_cost;
};

enum class CostFun {
	BinaryCrossentropy,
	CategoricalCrossentropy,
	MSE
};

class NeuralNetwork {
public:
	/**
	 * @brief Construct a new Neural Network.
	 * 
	 * @param input_layer Input layer of the NN.
	 * @param output_layer Output layer of the NN.
	 * @param cost_function Cost function used in gradient descent.
	 * @param cost_function_d Cost function derivative.
	 */
	NeuralNetwork(Layer& input_layer, Layer& output_layer, float(*cost_function)(const Tensor&, const Tensor&), const Tensor(*cost_function_d)(const Tensor&, const Tensor&));
	/**
	 * @brief Construct a new Neural Network.
	 * 
	 * @param input_layer Input layer of the NN.
	 * @param output_layer Output layer of the NN.
	 * @param cost_fun Cost function enum.
	 */
	NeuralNetwork(Layer& input_layer, Layer& output_layer, CostFun cost_fun);

	/**
	 * @brief Get the cost function.
	 * 
	 * @return NN cost function.
	 */
	float(*getCostFun())(const Tensor&, const Tensor&);
	/**
	 * @brief Performs inference on given data.
	 * 
	 * @param input Input data.
	 * @param inference Inference indicator (by default true).
	 * @return NN output values.
	 */
	const Tensor predict(const Tensor& input, bool inference=true);
	/**
	 * @brief Trains NN on given data.
	 * 
	 * @param train_x Train input data.
	 * @param train_y Train output data.
	 * @param test_x Test input data.
	 * @param test_y Test output data.
	 * @param batch_size Size of batch used in train algorithm.
	 * @param epochs Number of train epochs.
	 * @param learning_step Learning step parameter.
	 * @param momentum Momentum used to compute gradient velocity.
	 * @param verbose Verbose level indicator.
	 * @return FitHistory of performed training.
	 */
	FitHistory fit(const Tensor& train_x, const Tensor& train_y, const Tensor& test_x, const Tensor& test_y, uint32_t batch_size, uint32_t epochs, float learning_step, float momentum, uint8_t verbose=1u);

	/**
	 * @brief Print NN summary
	 */
	void summary() const;

	/**
	 * Cost functions and their derivatives.
	 */
	static float binary_crossentropy(const Tensor& y_hat, const Tensor& y);
	static const Tensor binary_crossentropy_d(const Tensor& y_hat, const Tensor& y);
	static float categorical_crossentropy(const Tensor& y_hat, const Tensor& y);
	static const Tensor categorical_crossentropy_d(const Tensor& y_hat, const Tensor& y);
	static float mse(const Tensor& y_hat, const Tensor& y);
	static const Tensor mse_d(const Tensor& y_hat, const Tensor& y);

private:
	/**
	 * Input layer of the NN.
	 */
	Layer* _input_layer;
	/**
	 * Output layer of the NN.
	 */
	Layer* _output_layer;
	/**
	 * Cost function used in gradient descent.
	 */
	float(*_cost_function)(const Tensor& y_hat, const Tensor& y);
	/**
	 * Cost function derivative.
	 */
	const Tensor (*_cost_function_d)(const Tensor& y_hat, const Tensor& y);

	/**
	 * @brief Updates weights of layers according to gradient velocity.
	 * 
	 * @param learning_step Learning step parameter.
	 * @param momentum Momentum used to compute gradient velocity.
	 */
	void updateLayersWeights(float learning_step, float momentum);
	/**
	 * @brief Inits gradients for all layers.
	 */
	void initLayersCachedGradient();

	/**
	 * @brief Print progress bar.
	 * 
	 * @param percent Progress value (0.0-1.0).
	 */
	static void print_progress(float progress);
	/**
	 * @brief Prints formatted time.
	 * 
	 * @param seconds Time in seconds.
	 */
	static void print_time(double seconds);
};