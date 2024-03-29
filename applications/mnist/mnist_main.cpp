#include "src/Tensor.h"
#include "src/NeuralNetwork.h"
#include "src/ActivationLayer.h"
#include "src/Pool2DLayer.h"
#include "src/Conv2DLayer.h"
#include "src/DenseLayer.h"
#include "src/ReshapeLayer.h"
#include "src/FlattenLayer.h"
#include "src/DropoutLayer.h"

#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include <ctime>
#include <cmath>
#include <iterator>
#include <sstream>
#include <string>
#include <unistd.h>

constexpr uint32_t train_data_len{ 60'000u };
constexpr uint32_t test_data_len{ 10'000u };
constexpr uint32_t image_size{ 28u };

static int read_data(const char* file_name, Tensor& data, Tensor& labels);

int main(int argc , char** argv) {
    // get this file dir
    char path[128] = __FILE__;

    // remove filename from path
    for (uint32_t i = 127; i >= 0; --i) {
        if ('/' == path[i]) {
            path[i] = 0;
            break;
        }
    }

    // change dir to this file dir
    if (chdir(path)) {
        std::cout << "Failed to change dir" << std::endl;
        return 1;
    }
    std::cout << "Changed dir to " << path << std::endl;

    // read data
    std::cout << "Reading data" << std::endl;
    Tensor train_data = Tensor({ train_data_len, image_size, image_size, 1 });
    Tensor train_labels = Tensor({ train_data_len, 10 });
    train_labels *= 0.0f;

    Tensor test_data = Tensor({ test_data_len, image_size, image_size, 1 });
    Tensor test_labels = Tensor({ test_data_len, 10 });
    test_labels *= 0.0f;

    if (read_data("./data/mnist_train.csv", train_data, train_labels)) {
        return 1;
    }
    
    if (read_data("./data/mnist_test.csv", test_data, test_labels)) {
        return 1;
    }
    std::cout << "Reading data done" << std::endl;

    // create neural network

    // dense model

    printf("Using dense model.\n");

    auto flatten = FlattenLayer({ 28, 28, 1});
    auto layer_dense_1 = DenseLayer(flatten, 512);
    auto layer_relu_1 = ActivationLayer(layer_dense_1, ActivationFun::LeakyReLU);
    auto dropout_1 = DropoutLayer(layer_relu_1, 0.3f);
    auto layer_dense_2 = DenseLayer(dropout_1, 512);
    auto layer_relu_4 = ActivationLayer(layer_dense_2, ActivationFun::LeakyReLU);
    auto dropout_2 = DropoutLayer(layer_relu_4, 0.3f);
    auto layer_dense_5 = DenseLayer(dropout_2, 10);
    auto layer_sigmoid = ActivationLayer(layer_dense_5, ActivationFun::Sigmoid);

    auto nn = NeuralNetwork(flatten, layer_sigmoid, CostFun::BinaryCrossentropy);

    // conv model

    // printf("Using conv model.\n");

    // // 28x28 -> 14x14
    // auto layer_conv2d_1 = Conv2DLayer({ 28, 28, 1 }, 32, 3);
    // auto layer_relu_1 = ActivationLayer(layer_conv2d_1, ActivationFun::LeakyReLU);
    // auto layer_pool2d_1 = Pool2DLayer(layer_relu_1, 2, PoolMode::Max);

    // // // 14x14 -> 7x7
    // // auto layer_conv2d_2 = Conv2DLayer(layer_pool2d_1, 16, 3);
    // // auto layer_relu_2 = ActivationLayer(layer_conv2d_2, ActivationFun::LeakyReLU);
    // // auto layer_pool2d_2 = Pool2DLayer(layer_relu_2, 2, PoolMode::Max);
    
    // // // 7x7 -> 7x7
    // // auto layer_conv2d_3 = Conv2DLayer(layer_pool2d_2, 32, 3);
    // // auto layer_relu_3 = ActivationLayer(layer_conv2d_3, ActivationFun::LeakyReLU);

    // // flatten
    // auto flatten = FlattenLayer(layer_pool2d_1);

    // // dense
    // auto layer_dense_1 = DenseLayer(flatten, 100);
    // auto layer_relu_4 = ActivationLayer(layer_dense_1, ActivationFun::LeakyReLU);
    // // auto layer_dense_2 = DenseLayer(layer_relu_4, 64);
    // // auto layer_relu_5 = ActivationLayer(layer_dense_2, ActivationFun::LeakyReLU);
    // auto layer_dense_3 = DenseLayer(layer_relu_4, 10);

    // auto layer_sigmoid = ActivationLayer(layer_dense_3, ActivationFun::Sigmoid);

    // auto nn = NeuralNetwork(layer_conv2d_1, layer_sigmoid, CostFun::BinaryCrossentropy);

    nn.summary();

    nn.fit(
        train_data, train_labels,
        test_data, test_labels,
        32,
        8,
        0.05f,
        0.8f);

    uint32_t valid_cnt{ 0 };

    constexpr uint32_t batch_size{ 4 };

    for (uint32_t i{ 0 }; (i + 1) <= (test_data.getShape()[0]/batch_size); ++i) {
        const Tensor batch_x = const_cast<const Tensor&>(test_data)[Tensor::Range({{ i*batch_size, (i + 1)*batch_size }})];
        const Tensor batch_y = const_cast<const Tensor&>(test_labels)[Tensor::Range({{ i*batch_size, (i + 1)*batch_size }})];

        const Tensor pred_label = nn.predict(batch_x);

        for (uint32_t j{ 0 }; j < batch_size; ++j) {
            float max_val{ -1.0f };
            uint32_t max_idx{ 0 };
            for (uint32_t k{ 0 }; k < pred_label.getShape()[1]; ++k)
            {
                if (max_val < 0 || max_val < pred_label[{ j, k }]) {
                    max_val = pred_label[{ j, k }];
                    max_idx = k;
                }
            }
            if (batch_y[{ j, max_idx }] > 0.0f) {
                ++valid_cnt;
            }
        }
    }

    printf("\nValid: %d / %d [%4.2f%%]\n", valid_cnt, test_data.getShape()[0], static_cast<float>(100*valid_cnt)/test_data.getShape()[0]);

}

int read_data(const char* file_name, Tensor& data, Tensor& labels) {
    std::ifstream train_file(file_name);
    if (train_file.is_open()) {
        std::string line;
        std::vector<uint32_t> numbers;
        for (uint32_t i = 0; i < data.getShape()[0]; ++i) {
            std::cout << "\r" << i + 1 << "/" << data.getShape()[0] << "   ";
            if (std::getline(train_file, line)) {
                std::replace(line.begin(), line.end(), ',', ' ');

                std::istringstream is(line);

                auto numbers = std::vector<uint32_t>(std::istream_iterator<uint32_t>(is),
                                                        std::istream_iterator<uint32_t>());
                
                labels[{ i, numbers[0] }] = 1.0f;

                for (uint32_t x = 0; x < image_size; ++x) {
                    for (uint32_t y = 0; y < image_size; ++y) {
                        data[{ i, x, y, 0 }] = static_cast<float>(numbers[1 + x*image_size + y])/255.0f;
                    }
                }
            }
            else {
                std::cout << "Wrong size of file" << std::endl;
                return 1;
            }
        }
        std::cout << "Done" << std::endl;
    }
    else {
        std::cout << "Could not open " << file_name << std::endl;
        return 1;
    }
    return 0;
}