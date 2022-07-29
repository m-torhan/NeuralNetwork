#include <gtest/gtest.h>
#include "src/ActivationLayer.h"
#include "tests/unit_tests/UnitTestsUtils.h"

TEST(ActivationLayer_test, ActivationLayerShouldApplyGivenFunctionWhenPrograpateForward) {
    Tensor tensor = Tensor({ 2, 2 });
    ActivationLayer layer = ActivationLayer({ 2, 2 }, [](const Tensor& x) -> const Tensor { return x * x; }, [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(result)[{ 0, 0 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(result)[{ 0, 1 }]));
    ASSERT_EQ( 9.0f, (const_cast<const Tensor&>(result)[{ 1, 0 }]));
    ASSERT_EQ(16.0f, (const_cast<const Tensor&>(result)[{ 1, 1 }]));
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

    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(result)[{ 0, 0 }]));
    ASSERT_EQ( 8.0f, (const_cast<const Tensor&>(result)[{ 0, 1 }]));
    ASSERT_EQ(18.0f, (const_cast<const Tensor&>(result)[{ 1, 0 }]));
    ASSERT_EQ(32.0f, (const_cast<const Tensor&>(result)[{ 1, 1 }]));
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

    const Tensor forward = layer.forwardPropagation(tensor);
    const Tensor backward = layer.backwardPropagation(tensor_back);

    ASSERT_EQ_EPS(0.26894f, forward[{ 0 }]);
    ASSERT_EQ_EPS(0.5f,     forward[{ 1 }]);
    ASSERT_EQ_EPS(0.73106f, forward[{ 2 }]);

    ASSERT_EQ_EPS(0.04915f, backward[{ 0 }]);
    ASSERT_EQ_EPS(0.1f,     backward[{ 1 }]);
    ASSERT_EQ_EPS(-0.2949f, backward[{ 2 }]);
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

    ASSERT_EQ_EPS(0.0f,  const_cast<const Tensor&>(forward)[{ 0 }]);
    ASSERT_EQ_EPS(0.25f, const_cast<const Tensor&>(forward)[{ 1 }]);
    ASSERT_EQ_EPS(1.0f,  const_cast<const Tensor&>(forward)[{ 2 }]);

    ASSERT_EQ_EPS(0.0f,  const_cast<const Tensor&>(backward)[{ 0 }]);
    ASSERT_EQ_EPS(0.4f,  const_cast<const Tensor&>(backward)[{ 1 }]);
    ASSERT_EQ_EPS(-1.5f, const_cast<const Tensor&>(backward)[{ 2 }]);
}