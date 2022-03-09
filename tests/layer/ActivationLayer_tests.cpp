#include <gtest/gtest.h>
#include "src/ActivationLayer.h"

TEST(ActivationLayer_test, ActivationLayerShouldApplyGivenFunctionWhenPrograpateForward) {
    Tensor tensor = Tensor({ 2, 2 });
    ActivationLayer layer = ActivationLayer({ 2, 2 }, [](const Tensor& x) -> const Tensor { return x * x; }, [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ( 1.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ( 4.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ( 9.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(16.0f, result.getValue({ 1, 1 }));
}

TEST(ActivationLayer_test, ActivationLayerShouldApplyGivenFunctionDerivativeWhenPrograpateBackward) {
    Tensor tensor = Tensor({ 2, 2 });
    ActivationLayer layer = ActivationLayer({ 2, 2 }, [](const Tensor& x) -> const Tensor { return x * x; }, [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    layer.forwardPropagation(tensor);
    Tensor result = layer.backwardPropagation(tensor);

    ASSERT_EQ( 2.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ( 8.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ(18.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(32.0f, result.getValue({ 1, 1 }));
}

TEST(ActivationLayer_test, SigmoidActivationValuesTest) {
    Tensor tensor = Tensor({ 3 });
    Tensor tensor_back = Tensor({ 3 });
    ActivationLayer layer = ActivationLayer({ 2, 2 }, ActivationFun::Sigmoid);

    tensor.setValues({
        -1.0f, 0.0f, 1.0f
        });

    tensor_back.setValues({
        0.25f, 0.4f, -1.5f
        });

    Tensor forward = layer.forwardPropagation(tensor);
    Tensor backward = layer.backwardPropagation(tensor_back);

    ASSERT_TRUE(fabs(0.26894f - forward.getValue({ 0 })) < 0.001f);
    ASSERT_TRUE(fabs(0.5f - forward.getValue({ 1 })) < 0.001f);
    ASSERT_TRUE(fabs(0.73106f - forward.getValue({ 2 })) < 0.001f);

    ASSERT_TRUE(fabs(0.04915f - backward.getValue({ 0 })) < 0.001f);
    ASSERT_TRUE(fabs(0.1f - backward.getValue({ 1 })) < 0.001f);
    ASSERT_TRUE(fabs(-0.29491f - backward.getValue({ 2 })) < 0.001f);
}

TEST(ActivationLayer_test, ReLUActivationValuesTest) {
    Tensor tensor = Tensor({ 3 });
    Tensor tensor_d = Tensor({ 3 });
    ActivationLayer layer = ActivationLayer({ 2, 2 }, ActivationFun::ReLU);

    tensor.setValues({
        -1.0f, 0.25f, 1.0f
        });

    tensor_d.setValues({
        0.25f, 0.4f, -1.5f
        });

    Tensor forward = layer.forwardPropagation(tensor);
    Tensor backward = layer.backwardPropagation(tensor_d);

    ASSERT_TRUE(fabs(0.0f - forward.getValue({ 0 })) < 0.001f);
    ASSERT_TRUE(fabs(0.25f - forward.getValue({ 1 })) < 0.001f);
    ASSERT_TRUE(fabs(1.0f - forward.getValue({ 2 })) < 0.001f);

    ASSERT_TRUE(fabs(0.0f - backward.getValue({ 0 })) < 0.001f);
    ASSERT_TRUE(fabs(0.4f - backward.getValue({ 1 })) < 0.001f);
    ASSERT_TRUE(fabs(-1.5f - backward.getValue({ 2 })) < 0.001f);
}