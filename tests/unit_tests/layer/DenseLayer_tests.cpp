#include <gtest/gtest.h>
#include "src/DenseLayer.h"
#include "tests/unit_tests/UnitTestsUtils.h"

TEST(DenseLayer_test, DenseLayerForwardPropagationOutputShapeTest) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    DenseLayer layer = DenseLayer({ 3, 4 }, 4);

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(4, (int)result.getShape()[1]);
}

TEST(DenseLayer_test, DenseLayerBackwardPropagationOutputShapeTest) {
    Tensor tensor = Tensor({ 2, 12 });
    Tensor tensor_d = Tensor({ 2, 4 });
    DenseLayer layer = DenseLayer({ 3, 4 }, 4);

    layer.initCachedGradient();
    layer.forwardPropagation(tensor, false);
    Tensor result = layer.backwardPropagation(tensor_d);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(12, (int)result.getShape()[1]);
}

TEST(DenseLayer_test, DenseLayerForwardAndBackwardPropagationReturnValuesTest) {
    Tensor tensor = Tensor({ 2, 3 });
    Tensor tensor_d = Tensor({ 2, 2 });
    DenseLayer layer = DenseLayer({ 3 }, 2);

    layer.initCachedGradient();

    tensor.setValues({
        1.0f, -0.5f, 2.2f,
        3.1f, -10.0f, 123.0f
        });

    tensor_d.setValues({
        -1.0f, -1.5f,
        1.7f, 43.2f
        });

    layer.setWeights({
        -0.2f, 0.123f, 0.43f,
        -0.543f, 0.2433f, -0.654f
        });
    layer.setBiases({
        0.0f, -0.5f
        });

    const Tensor forward = layer.forwardPropagation(tensor, false);
    const Tensor backward = layer.backwardPropagation(tensor_d);

    ASSERT_EQ_EPS(  0.6845f,  (forward[{ 0, 0 }]));
    ASSERT_EQ_EPS( -2.60345f, (forward[{ 0, 1 }]));
    ASSERT_EQ_EPS( 51.04f,    (forward[{ 1, 0 }]));
    ASSERT_EQ_EPS(-85.0583f,  (forward[{ 1, 1 }]));

    ASSERT_EQ_EPS(  1.0145f,  (backward[{ 0, 0 }]));
    ASSERT_EQ_EPS( -0.48795f, (backward[{ 0, 1 }]));
    ASSERT_EQ_EPS(  0.551f,   (backward[{ 0, 2 }]));
    ASSERT_EQ_EPS(-23.7976f,  (backward[{ 1, 0 }]));
    ASSERT_EQ_EPS( 10.71966f, (backward[{ 1, 1 }]));
    ASSERT_EQ_EPS(-27.5218f,  (backward[{ 1, 2 }]));
}