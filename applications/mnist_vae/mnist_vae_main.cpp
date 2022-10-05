#include "src/Tensor.h"
#include "src/NeuralNetwork.h"
#include "src/ActivationLayer.h"
#include "src/Pool2DLayer.h"
#include "src/Conv2DLayer.h"
#include "src/DenseLayer.h"
#include "src/ReshapeLayer.h"
#include "src/NormalDistLayer.h"
#include "src/FlattenLayer.h"

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

constexpr uint32_t train_data_len{ 6'000u };
constexpr uint32_t test_data_len{ 1'000u };
constexpr uint32_t image_size{ 28u };

static int read_data(const char* file_name, Tensor& data, Tensor& labels);
static void show_digit(const Tensor& nn_output);

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

    if (read_data("../mnist/data/mnist_train.csv", train_data, train_labels)) {
        return 1;
    }
    
    if (read_data("../mnist/data/mnist_test.csv", test_data, test_labels)) {
        return 1;
    }
    std::cout << "Reading data done" << std::endl;

    // for (uint32_t i{ 0 }; i < 10; ++i) {
    //     show_digit(const_cast<const Tensor&>(train_data)[Tensor::Range({{ i }})]);
    // }

    // create neural network

    // dense model

    printf("Using dense model.\n");

    constexpr uint32_t latent_dim{ 2 };
    constexpr uint32_t neurons_count{ 16 };

    // encoder
    auto enc_flatten = FlattenLayer({ 28, 28, 1 });
    auto enc_layer_dense_1 = DenseLayer(enc_flatten, neurons_count * 16);
    // auto enc_layer_relu_1 = ActivationLayer(enc_layer_dense_1, ActivationFun::LeakyReLU);
    // auto enc_layer_dense_2 = DenseLayer(enc_layer_relu_1, 1024);
    auto enc_layer_relu_2 = ActivationLayer(enc_layer_dense_1, ActivationFun::LeakyReLU);
    auto enc_layer_dense_3 = DenseLayer(enc_layer_relu_2, neurons_count * 8);
    auto enc_layer_relu_3 = ActivationLayer(enc_layer_dense_3, ActivationFun::LeakyReLU);
    auto enc_layer_dense_4 = DenseLayer(enc_layer_relu_3, neurons_count * 4);
    auto enc_layer_relu_4 = ActivationLayer(enc_layer_dense_4, ActivationFun::LeakyReLU);
    auto enc_layer_dense_5 = DenseLayer(enc_layer_relu_4, neurons_count * 2);
    auto enc_layer_relu_5 = ActivationLayer(enc_layer_dense_5, ActivationFun::LeakyReLU);
    auto enc_layer_dense_6 = DenseLayer(enc_layer_relu_5, neurons_count);
    auto enc_layer_relu_6 = ActivationLayer(enc_layer_dense_6, ActivationFun::LeakyReLU);
    auto enc_layer_dense_7 = DenseLayer(enc_layer_relu_6, latent_dim * 2);

    auto enc_layer_reshape_1 = ReshapeLayer(enc_layer_dense_7, { latent_dim, 2 });
    auto enc_norm_dist_layer = NormalDistLayer(enc_layer_reshape_1);
    auto enc_layer_reshape_2 = ReshapeLayer(enc_norm_dist_layer, { latent_dim });

    //decoder
    auto dec_layer_dense_1 = DenseLayer(enc_layer_reshape_2, neurons_count);
    auto dec_layer_relu_1 = ActivationLayer(dec_layer_dense_1, ActivationFun::LeakyReLU);
    auto dec_layer_dense_2 = DenseLayer(dec_layer_relu_1, neurons_count * 2);
    auto dec_layer_relu_2 = ActivationLayer(dec_layer_dense_2, ActivationFun::LeakyReLU);
    auto dec_layer_dense_3 = DenseLayer(dec_layer_relu_2, neurons_count * 4);
    auto dec_layer_relu_3 = ActivationLayer(dec_layer_dense_3, ActivationFun::LeakyReLU);
    auto dec_layer_dense_4 = DenseLayer(dec_layer_relu_3, neurons_count * 8);
    auto dec_layer_relu_4 = ActivationLayer(dec_layer_dense_4, ActivationFun::LeakyReLU);
    auto dec_layer_dense_5 = DenseLayer(dec_layer_relu_4, neurons_count * 16);
    auto dec_layer_relu_5 = ActivationLayer(dec_layer_dense_5, ActivationFun::LeakyReLU);
    // auto dec_layer_dense_5 = DenseLayer(dec_layer_relu_4, 1024);
    // auto dec_layer_relu_5 = ActivationLayer(dec_layer_dense_5, ActivationFun::LeakyReLU);
    auto dec_layer_dense_6 = DenseLayer(dec_layer_relu_5, 784);

    auto dec_layer_sigmoid = ActivationLayer(dec_layer_dense_6, ActivationFun::Sigmoid);
    auto dec_layer_reshape = ReshapeLayer(dec_layer_sigmoid, { 28, 28, 1});

    auto vae = NeuralNetwork(enc_flatten, dec_layer_reshape, CostFun::BinaryCrossentropy);
    auto enc = NeuralNetwork(enc_flatten, enc_layer_reshape_2, CostFun::BinaryCrossentropy);
    auto gen = NeuralNetwork(dec_layer_dense_1, dec_layer_reshape, CostFun::BinaryCrossentropy);

    // conv model

    // printf("Using conv model.");

    // // 28x28 -> 14x14
    // auto layer_conv2d_1 = Conv2DLayer({ 28, 28, 1 }, 4, 3);
    // auto layer_relu_1 = ActivationLayer(layer_conv2d_1, ActivationFun::LeakyReLU);
    // auto layer_pool2d_1 = Pool2DLayer(layer_relu_1, 2, PoolMode::Max);

    // // 14x14 -> 7x7
    // auto layer_conv2d_2 = Conv2DLayer(layer_pool2d_1, 8, 3);
    // auto layer_relu_2 = ActivationLayer(layer_conv2d_2, ActivationFun::LeakyReLU);
    // auto layer_pool2d_2 = Pool2DLayer(layer_relu_2, 2, PoolMode::Max);
    
    // // 7x7 -> 7x7
    // auto layer_conv2d_3 = Conv2DLayer(layer_pool2d_2, 16, 3);
    // auto layer_relu_3 = ActivationLayer(layer_conv2d_3, ActivationFun::LeakyReLU);

    // // dense
    // auto layer_dense_1 = DenseLayer(layer_relu_3, 32);
    // auto layer_relu_4 = ActivationLayer(layer_dense_1, ActivationFun::LeakyReLU);
    // auto layer_dense_2 = DenseLayer(layer_relu_4, 10);

    // auto layer_sigmoid = ActivationLayer(layer_dense_2, ActivationFun::Sigmoid);

    // auto vae = NeuralNetwork(layer_conv2d_1, layer_sigmoid, CostFun::BinaryCrossentropy);

    vae.summary();

    vae.fit(
        train_data, train_data,
        test_data, test_data,
        64,
        32,
        0.001f,
        0.8f);

    Tensor random_tensor = Tensor::RandomNormal({ 10, latent_dim });

    const Tensor output = gen.predict(random_tensor);

    for (uint32_t i{ 0 }; i < 10; ++i) {
        Tensor img = output[Tensor::Range({{ i }})];
        img -= img.min();
        img /= img.max();
        show_digit(img);
    }

    // encoding digits from test set

    constexpr uint32_t batch_size{ 10 };

    std::ofstream encoded_test_file("./data/test_encoded.txt");

    for (uint32_t i{ 0 }; (i + 1) <= (test_data.getShape()[0]/batch_size); ++i) {
        const Tensor batch_x = const_cast<const Tensor&>(test_data)[Tensor::Range({{ i*batch_size, (i + 1)*batch_size }})];
        const Tensor batch_y = const_cast<const Tensor&>(test_labels)[Tensor::Range({{ i*batch_size, (i + 1)*batch_size }})];

        const Tensor encoded = enc.predict(batch_x);

        for (uint32_t j{ 0 }; j < batch_size; ++j) {
            float max_val{ -1.0f };
            uint32_t max_idx{ 0 };

            for (uint32_t k{ 0 }; k < batch_y.getShape()[1]; ++k)
            {
                if (max_val < 0 || max_val < batch_y[{ j, k }]) {
                    max_val = batch_y[{ j, k }];
                    max_idx = k;
                }
            }
            encoded_test_file << max_idx << ";" << encoded[{ j, 0 }] << ";" << encoded[{ j, 1 }] << std::endl;
        }
    }

    encoded_test_file.close();
    
    // decoding values along selected line from (-0.8, -0.7) to (0.7, 0.8)

    std::ofstream generated_file("./data/gen_along_line.txt");

    constexpr uint32_t line_segments = 10000;
    Tensor line_input({ line_segments, latent_dim });

    Tensor start_point({ 2 });
    Tensor end_point({ 2 });

    start_point[{ 0 }] = -0.8;
    start_point[{ 1 }] = -0.7;

    end_point[{ 0 }] = 0.7;
    end_point[{ 1 }] = 0.8;

    for (uint32_t i{ 0 }; i < line_segments; ++i) {
        float p = static_cast<float>(i)/line_segments;
        line_input[{ i, 0 }] = const_cast<const Tensor&>(start_point)[{ 0 }]*p + const_cast<const Tensor&>(end_point)[{ 0 }]*(1 - p);
        line_input[{ i, 1 }] = const_cast<const Tensor&>(start_point)[{ 1 }]*p + const_cast<const Tensor&>(end_point)[{ 1 }]*(1 - p);
    }

    for (uint32_t i{ 0 }; (i + 1) <= (line_input.getShape()[0]/batch_size); ++i) {
        const Tensor batch_line = const_cast<const Tensor&>(line_input)[Tensor::Range({{ i*batch_size, (i + 1)*batch_size }})];

        const Tensor generated = gen.predict(batch_line);

        for (uint32_t j{ 0 }; j < batch_size; ++j) {
            generated_file << batch_line[{ j, 0 }] << ";" << batch_line[{ j, 1 }];
            for (uint32_t x{ 0 }; x < image_size; ++x) { 
                for (uint32_t y{ 0 }; y < image_size; ++y) {
                    generated_file << ";" << generated[{ j, x, y, 0 }];
                }
            }
            generated_file << std::endl;
        }
    }

    generated_file.close();
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
    train_file.close();
    return 0;
}

static void show_digit(const Tensor& nn_output) {
    printf("============================\n");
    for (uint32_t i{ 0 }; i < nn_output.getShape()[0]; ++i) {
        for (uint32_t j{ 0 }; j < nn_output.getShape()[1]; ++j) {
            if (nn_output[{ i, j, 0}] > 0.8) printf("#");
            else if (nn_output[{ i, j, 0}] > 0.5) printf("x");
            else if (nn_output[{ i, j, 0}] > 0.3) printf(".");
            else printf(" ");
        }
        printf("\n");
    }
    printf("============================\n\n");
}