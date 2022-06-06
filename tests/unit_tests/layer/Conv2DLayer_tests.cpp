#include <gtest/gtest.h>
#include "src/Conv2DLayer.h"
#include "tests/unit_tests/UnitTestsUtils.h"

TEST(Conv2DLayer_test, Conv2DLayerForwardPropagationOutputShapeTest) {
    Tensor tensor = Tensor({ 2, 3, 4, 5 });
    Conv2DLayer layer = Conv2DLayer({ 3, 4, 5 }, 6, 3);

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ(4, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);
    ASSERT_EQ(4, (int)result.getShape()[2]);
    ASSERT_EQ(6, (int)result.getShape()[3]);
}

TEST(Conv2DLayer_test, Conv2DLayerBackwardPropagationOutputShapeTest) {
    Tensor tensor = Tensor({ 2, 3, 4, 5 });
    Tensor tensor_d = Tensor({ 2, 3, 4, 6 });
    Conv2DLayer layer = Conv2DLayer({ 3, 4, 5 }, 6, 3);

    layer.initCachedGradient();
    layer.forwardPropagation(tensor);
    Tensor result = layer.backwardPropagation(tensor_d);

    ASSERT_EQ(4, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);
    ASSERT_EQ(4, (int)result.getShape()[2]);
    ASSERT_EQ(5, (int)result.getShape()[3]);
}

TEST(Conv2DLayer_test, Conv2DLayerForwardPropagationReturnValuesTest) {
    Tensor tensor = Tensor({ 1, 3, 4, 2 });
    Tensor tensor_d = Tensor({ 1, 3, 4, 3 });
    Conv2DLayer layer = Conv2DLayer({ 3, 4, 2 }, 3, 3);
    
    layer.initCachedGradient();

    tensor.setValues({
         1.0f,  2.0f,   3.0f,  4.0f,   5.0f,  6.0f,   7.0f,  8.0f,
         9.0f, 10.0f,  11.0f, 12.0f,  13.0f, 14.0f,  15.0f, 16.0f,
        17.0f, 18.0f,  19.0f, 20.0f,  21.0f, 22.0f,  23.0f, 24.0f
    });
    
    tensor_d.setValues({
         1.0f,  2.0f,  3.0f,   4.0f,  5.0f,  6.0f,   7.0f,  8.0f,  9.0f,  10.0f, -9.0f, -8.0f,
        -7.0f, -6.0f, -5.0f,   0.0f,  0.5f,  0.0f,   0.0f,  1.0f,  0.0f,   0.5f,  0.0f,  0.0f,
        -4.0f, -3.0f, -2.0f,  -1.0f,  0.0f,  0.5f,   0.0f,  5.0f,  0.5f,   0.0f,  0.0f,  1.0f
    });

     layer.setWeights({ 
         1.0f,  0.0f,  0.0f,   0.0f,  0.0f,  0.0f,     0.0f,  0.0f,  0.0f,  0.0f,  2.0f,  0.0f,     0.0f,  0.0f, -3.0f,   0.0f,  0.0f, -1.0f,
         0.0f,  0.0f,  0.0f,   0.0f,  0.0f,  0.0f,     1.0f,  0.0f, -3.0f,  0.0f,  2.0f, -1.0f,     0.0f,  0.0f,  0.0f,   0.0f,  0.0f,  0.0f,
         0.0f,  0.0f, -3.0f,   0.0f,  0.0f, -1.0f,     0.0f,  0.0f,  0.0f,  0.0f,  2.0f,  0.0f,     1.0f,  0.0f,  0.0f,   0.0f,  0.0f,  0.0f
    });

    layer.setBiases({
        0.0f, 1.0f, -2.0f
    });

    Tensor forward = layer.forwardPropagation(tensor);
    Tensor backward = layer.backwardPropagation(tensor_d);

    ASSERT_EQ_EPS(  12.0f, forward.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ_EPS(  25.0f, forward.getValue({ 0, 0, 0, 1 }));
    ASSERT_EQ_EPS(  -7.0f, forward.getValue({ 0, 0, 0, 2 }));
    ASSERT_EQ_EPS(  16.0f, forward.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ_EPS(  33.0f, forward.getValue({ 0, 0, 1, 1 }));
    ASSERT_EQ_EPS( -52.0f, forward.getValue({ 0, 0, 1, 2 }));
    ASSERT_EQ_EPS(  20.0f, forward.getValue({ 0, 0, 2, 0 }));
    ASSERT_EQ_EPS(  41.0f, forward.getValue({ 0, 0, 2, 1 }));
    ASSERT_EQ_EPS( -68.0f, forward.getValue({ 0, 0, 2, 2 }));
    ASSERT_EQ_EPS(   7.0f, forward.getValue({ 0, 0, 3, 0 }));
    ASSERT_EQ_EPS(  49.0f, forward.getValue({ 0, 0, 3, 1 }));
    ASSERT_EQ_EPS( -84.0f, forward.getValue({ 0, 0, 3, 2 }));
    ASSERT_EQ_EPS(  28.0f, forward.getValue({ 0, 1, 0, 0 }));
    ASSERT_EQ_EPS(  61.0f, forward.getValue({ 0, 1, 0, 1 }));
    ASSERT_EQ_EPS( -52.0f, forward.getValue({ 0, 1, 0, 2 }));
    ASSERT_EQ_EPS(  33.0f, forward.getValue({ 0, 1, 1, 0 }));
    ASSERT_EQ_EPS(  73.0f, forward.getValue({ 0, 1, 1, 1 }));
    ASSERT_EQ_EPS(-137.0f, forward.getValue({ 0, 1, 1, 2 }));
    ASSERT_EQ_EPS(  39.0f, forward.getValue({ 0, 1, 2, 0 }));
    ASSERT_EQ_EPS(  85.0f, forward.getValue({ 0, 1, 2, 1 }));
    ASSERT_EQ_EPS(-161.0f, forward.getValue({ 0, 1, 2, 2 }));
    ASSERT_EQ_EPS(  20.0f, forward.getValue({ 0, 1, 3, 0 }));
    ASSERT_EQ_EPS(  97.0f, forward.getValue({ 0, 1, 3, 1 }));
    ASSERT_EQ_EPS(-148.0f, forward.getValue({ 0, 1, 3, 2 }));
    ASSERT_EQ_EPS(  17.0f, forward.getValue({ 0, 2, 0, 0 }));
    ASSERT_EQ_EPS(  57.0f, forward.getValue({ 0, 2, 0, 1 }));
    ASSERT_EQ_EPS(-116.0f, forward.getValue({ 0, 2, 0, 2 }));
    ASSERT_EQ_EPS(  28.0f, forward.getValue({ 0, 2, 1, 0 }));
    ASSERT_EQ_EPS(  65.0f, forward.getValue({ 0, 2, 1, 1 }));
    ASSERT_EQ_EPS(-132.0f, forward.getValue({ 0, 2, 1, 2 }));
    ASSERT_EQ_EPS(  32.0f, forward.getValue({ 0, 2, 2, 0 }));
    ASSERT_EQ_EPS(  73.0f, forward.getValue({ 0, 2, 2, 1 }));
    ASSERT_EQ_EPS(-148.0f, forward.getValue({ 0, 2, 2, 2 }));
    ASSERT_EQ_EPS(  36.0f, forward.getValue({ 0, 2, 3, 0 }));
    ASSERT_EQ_EPS(  81.0f, forward.getValue({ 0, 2, 3, 1 }));
    ASSERT_EQ_EPS( -95.0f, forward.getValue({ 0, 2, 3, 2 }));

    ASSERT_EQ_EPS( -8.0f, backward.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ_EPS(-11.0f, backward.getValue({ 0, 0, 0, 1 }));
    ASSERT_EQ_EPS(  1.0f, backward.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ_EPS( 10.0f, backward.getValue({ 0, 0, 1, 1 }));
    ASSERT_EQ_EPS(-19.5f, backward.getValue({ 0, 0, 2, 0 }));
    ASSERT_EQ_EPS(  9.0f, backward.getValue({ 0, 0, 2, 1 }));
    ASSERT_EQ_EPS( 34.0f, backward.getValue({ 0, 0, 3, 0 }));
    ASSERT_EQ_EPS(-10.0f, backward.getValue({ 0, 0, 3, 1 }));
    ASSERT_EQ_EPS(-11.0f, backward.getValue({ 0, 1, 0, 0 }));
    ASSERT_EQ_EPS(-15.0f, backward.getValue({ 0, 1, 0, 1 }));
    ASSERT_EQ_EPS(-20.0f, backward.getValue({ 0, 1, 1, 0 }));
    ASSERT_EQ_EPS(  4.0f, backward.getValue({ 0, 1, 1, 1 }));
    ASSERT_EQ_EPS( 26.5f, backward.getValue({ 0, 1, 2, 0 }));
    ASSERT_EQ_EPS( 35.5f, backward.getValue({ 0, 1, 2, 1 }));
    ASSERT_EQ_EPS(  6.0f, backward.getValue({ 0, 1, 3, 0 }));
    ASSERT_EQ_EPS(-18.5f, backward.getValue({ 0, 1, 3, 1 }));
    ASSERT_EQ_EPS(  2.0f, backward.getValue({ 0, 2, 0, 0 }));
    ASSERT_EQ_EPS(-16.0f, backward.getValue({ 0, 2, 0, 1 }));
    ASSERT_EQ_EPS( -9.5f, backward.getValue({ 0, 2, 1, 0 }));
    ASSERT_EQ_EPS(  0.5f, backward.getValue({ 0, 2, 1, 1 }));
    ASSERT_EQ_EPS( -1.5f, backward.getValue({ 0, 2, 2, 0 }));
    ASSERT_EQ_EPS( 11.5f, backward.getValue({ 0, 2, 2, 1 }));
    ASSERT_EQ_EPS( -3.0f, backward.getValue({ 0, 2, 3, 0 }));
    ASSERT_EQ_EPS( -1.0f, backward.getValue({ 0, 2, 3, 1 }));
}