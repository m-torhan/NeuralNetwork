#include <gtest/gtest.h>
#include "src/NeuralNetwork.h"
#include "src/ActivationLayer.h"
#include "src/DenseLayer.h"
#include "tests/unit_tests/UnitTestsUtils.h"

TEST(NeuralNetwork_test, BinaryCrossentropyTest) {
    Tensor y = Tensor({ 2, 2 });
    Tensor y_hat = Tensor({ 2, 2 });

    y.setValues({
        1.0f, 0.123f,
        0.2f, 0.45f
        });

    y_hat.setValues({
        0.2f, 0.123f,
        0.456f, 0.321f
        });

    float result = NeuralNetwork::binary_crossentropy(y_hat, y);
    const Tensor result_d = NeuralNetwork::binary_crossentropy_d(y_hat, y);

    ASSERT_TRUE(fabs(0.83766f - result) < 0.001f);

    ASSERT_EQ(2, (int)result_d.getDim());
    ASSERT_EQ(2, (int)result_d.getShape()[0]);
    ASSERT_EQ(2, (int)result_d.getShape()[1]);

    ASSERT_EQ_EPS(-4.99999f, (result_d[{ 0, 0 }]));
    ASSERT_EQ_EPS(0.0f,      (result_d[{ 0, 1 }]));
    ASSERT_EQ_EPS(1.03199f,  (result_d[{ 1, 0 }]));
    ASSERT_EQ_EPS(-0.59185f, (result_d[{ 1, 1 }]));
}

TEST(NeuralNetwork_test, PredictShouldReturnTensor) {
    Tensor tensor = Tensor({ 2, 2 });
    const Tensor (*activation_fun)(const Tensor & x) = [](const Tensor& x) -> const Tensor { return x * x; };
    const Tensor (*activation_fun_d)(const Tensor & x, const Tensor & dx) = [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); };
    ActivationLayer layer_1 = ActivationLayer({ 2, 2 }, activation_fun, activation_fun_d);
    ActivationLayer layer_2 = ActivationLayer(layer_1, activation_fun, activation_fun_d);
    ActivationLayer layer_3 = ActivationLayer(layer_2, activation_fun, activation_fun_d);

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, -1.0f
        });

    NeuralNetwork nn = NeuralNetwork(layer_1, layer_3, nullptr, nullptr);

    const Tensor result = nn.predict(tensor);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(2, (int)result.getShape()[1]);

    ASSERT_EQ(   1.0f, (result[{ 0, 0 }]));
    ASSERT_EQ( 256.0f, (result[{ 0, 1 }]));
    ASSERT_EQ(6561.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(   1.0f, (result[{ 1, 1 }]));
}

TEST(NeuralNetwork_test, FitShouldDecreaseCost) {
    auto x_train = Tensor({ 512, 2 });
    auto y_train = Tensor({ 512, 2 });
    auto x_test = Tensor({ 32, 2 });
    auto y_test = Tensor({ 32, 2 });

    srand(time(NULL));

    for (uint32_t i = 0; i < x_train.getShape()[0]; ++i) {
        float x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
        float y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

        x_train[{ i, 0 }] = x;
        x_train[{ i, 1 }] = y;

        float u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

        y_train[{ i, 0 }] = u;
        y_train[{ i, 1 }] = 1.0f - u;
    }
    for (uint32_t i = 0; i < x_test.getShape()[0]; ++i) {
        float x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
        float y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

        x_test[{ i, 0 }] = x;
        x_test[{ i, 1 }] = y;

        float u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

        y_test[{ i, 0 }] = u;
        y_test[{ i, 1 }] = 1.0f - u;
    }

    auto layer_1 = ActivationLayer({ 2 }, ActivationFun::ReLU);
    auto layer_2 = DenseLayer(layer_1, 16);
    auto layer_3 = ActivationLayer(layer_2, ActivationFun::ReLU);
    auto layer_4 = DenseLayer(layer_3, 16);
    auto layer_5 = ActivationLayer(layer_4, ActivationFun::ReLU);
    auto layer_6 = DenseLayer(layer_5, 2);
    auto layer_7 = ActivationLayer(layer_6, ActivationFun::Sigmoid);

    auto nn = NeuralNetwork(layer_1, layer_7, CostFun::BinaryCrossentropy);

    Tensor y_hat = nn.predict(x_test);

    float cost_1 = nn.getCostFun()(y_hat, y_test);

    nn.fit(x_train, y_train, x_test, y_test, 32, 10, 0.01f, 0);

    y_hat = nn.predict(x_test);

    float cost_2 = nn.getCostFun()(y_hat, y_test);

    ASSERT_TRUE(cost_2 < cost_1);
}