#include <gtest/gtest.h>
#include "src/Pool2DLayer.h"
#include "tests/unit_tests/UnitTestsUtils.h"

TEST(Pool2DLayer_test, Pool2DLayerAverageForwardPropagation) {
    Tensor tensor = Tensor({ 2, 4, 4, 1 });
    Pool2DLayer layer = Pool2DLayer({ 4, 4 }, 2, PoolMode::Average);

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,

        -2.0f,  0.0f, 19.0f, 20.0f,
        21.0f, -1.0f, 15.0f, -5.0f,
        13.0f,  7.0f,  1.0f, 16.0f,
         8.0f, 14.0f, 15.0f, -3.0f
        });

    Tensor result = layer.forwardPropagation(tensor);
    
    ASSERT_EQ( 4, result.getDim());
    ASSERT_EQ( 2, result.getShape()[0]);
    ASSERT_EQ( 2, result.getShape()[1]);
    ASSERT_EQ( 2, result.getShape()[2]);
    ASSERT_EQ( 1, result.getShape()[3]);

    ASSERT_EQ(   3.5f, result.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ(   5.5f, result.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ(  11.5f, result.getValue({ 0, 1, 0, 0 }));
    ASSERT_EQ(  13.5f, result.getValue({ 0, 1, 1, 0 }));
    ASSERT_EQ(   4.5f, result.getValue({ 1, 0, 0, 0 }));
    ASSERT_EQ( 12.25f, result.getValue({ 1, 0, 1, 0 }));
    ASSERT_EQ(  10.5f, result.getValue({ 1, 1, 0, 0 }));
    ASSERT_EQ(  7.25f, result.getValue({ 1, 1, 1, 0 }));
}

TEST(Pool2DLayer_test, Pool2DLayerAverageBackwardPropagation) {
    Tensor tensor = Tensor({ 2, 4, 4, 1 });
    Tensor tensor_d = Tensor({ 2, 2, 2, 1 });
    Pool2DLayer layer = Pool2DLayer({ 4, 4 }, 2, PoolMode::Average);

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,

        -2.0f,  0.0f, 19.0f, 20.0f,
        21.0f, -1.0f, 15.0f, -5.0f,
        13.0f,  7.0f,  1.0f, 16.0f,
         8.0f, 14.0f, 15.0f, -3.0f
        });
    
    tensor_d.setValues({
         7.0f, 11.0f,
        -1.0f,  0.0f,

         4.0f,  2.5f,
        10.0f, -3.0f,
    });

    layer.forwardPropagation(tensor);
    Tensor result = layer.backwardPropagation(tensor_d);
    
    ASSERT_EQ( 4, result.getDim());
    ASSERT_EQ( 2, result.getShape()[0]);
    ASSERT_EQ( 4, result.getShape()[1]);
    ASSERT_EQ( 4, result.getShape()[2]);
    ASSERT_EQ( 1, result.getShape()[3]);
    
    ASSERT_EQ_EPS(     2.0f, result.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ_EPS(     4.0f, result.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ_EPS(     6.0f, result.getValue({ 0, 0, 2, 0 }));
    ASSERT_EQ_EPS(     8.0f, result.getValue({ 0, 0, 3, 0 }));
    ASSERT_EQ_EPS( -1.7778f, result.getValue({ 1, 0, 0, 0 }));
    ASSERT_EQ_EPS(     0.0f, result.getValue({ 1, 0, 1, 0 }));
    ASSERT_EQ_EPS(  3.8775f, result.getValue({ 1, 0, 2, 0 }));
    ASSERT_EQ_EPS(  4.0816f, result.getValue({ 1, 0, 3, 0 }));
    ASSERT_EQ_EPS(  7.6191f, result.getValue({ 1, 3, 0, 0 }));
    ASSERT_EQ_EPS( 13.3333f, result.getValue({ 1, 3, 1, 0 }));
    ASSERT_EQ_EPS( -6.2069f, result.getValue({ 1, 3, 2, 0 }));
    ASSERT_EQ_EPS(  1.2414f, result.getValue({ 1, 3, 3, 0 }));
}

TEST(Pool2DLayer_test, Pool2DLayerMaxForwardPropagation) {
    Tensor tensor = Tensor({ 2, 4, 4, 1 });
    Pool2DLayer layer = Pool2DLayer({ 4, 4 }, 2, PoolMode::Max);

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,

        -2.0f,  0.0f, 19.0f, 20.0f,
        21.0f, -1.0f, 15.0f, -5.0f,
        13.0f,  7.0f,  1.0f, 16.0f,
         8.0f, 14.0f, 15.0f, -3.0f
        });

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ( 4, result.getDim());
    ASSERT_EQ( 2, result.getShape()[0]);
    ASSERT_EQ( 2, result.getShape()[1]);
    ASSERT_EQ( 2, result.getShape()[2]);
    ASSERT_EQ( 1, result.getShape()[3]);

    ASSERT_EQ(  6.0f, result.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ(  8.0f, result.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ( 14.0f, result.getValue({ 0, 1, 0, 0 }));
    ASSERT_EQ( 16.0f, result.getValue({ 0, 1, 1, 0 }));
    ASSERT_EQ( 21.0f, result.getValue({ 1, 0, 0, 0 }));
    ASSERT_EQ( 20.0f, result.getValue({ 1, 0, 1, 0 }));
    ASSERT_EQ( 14.0f, result.getValue({ 1, 1, 0, 0 }));
    ASSERT_EQ( 16.0f, result.getValue({ 1, 1, 1, 0 }));
}

TEST(Pool2DLayer_test, Pool2DLayerMaxBackwardPropagation) {
    Tensor tensor = Tensor({ 2, 4, 4, 1 });
    Tensor tensor_d = Tensor({ 2, 2, 2, 1 });
    Pool2DLayer layer = Pool2DLayer({ 4, 4 }, 2, PoolMode::Max);

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,
        13.0f, 14.0f, 15.0f, 16.0f,

        -2.0f,  0.0f, 19.0f, 20.0f,
        21.0f, -1.0f, 15.0f, -5.0f,
        13.0f,  7.0f,  1.0f, 16.0f,
         8.0f, 14.0f, 15.0f, -3.0f
        });
    
    tensor_d.setValues({
         7.0f, 11.0f,
        -1.0f,  0.0f,

         4.0f,  2.5f,
        10.0f, -3.0f,
    });

    layer.forwardPropagation(tensor);
    Tensor result = layer.backwardPropagation(tensor_d);
    
    ASSERT_EQ( 4, result.getDim());
    ASSERT_EQ( 2, result.getShape()[0]);
    ASSERT_EQ( 4, result.getShape()[1]);
    ASSERT_EQ( 4, result.getShape()[2]);
    ASSERT_EQ( 1, result.getShape()[3]);

    ASSERT_EQ_EPS(  0.0f, result.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 0, 0, 2, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 0, 0, 3, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 0, 0, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 0, 1, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 0, 2, 0 }));
    ASSERT_EQ_EPS(  2.5f, result.getValue({ 1, 0, 3, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 3, 0, 0 }));
    ASSERT_EQ_EPS( 10.0f, result.getValue({ 1, 3, 1, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 3, 2, 0 }));
    ASSERT_EQ_EPS(  0.0f, result.getValue({ 1, 3, 3, 0 }));
}

TEST(Pool2DLayer_test, Pool2DLayerMaxForwardPropagationTensorWithChannels) {
    Tensor tensor = Tensor({ 2, 4, 4, 2 });
    Pool2DLayer layer = Pool2DLayer({ 4, 4, 2 }, 2, PoolMode::Max);

    tensor.setValues({
         1.0f,  2.0f,    3.0f,  4.0f,    5.0f,  6.0f,    7.0f,  8.0f,
         9.0f, 10.0f,   11.0f, 12.0f,   13.0f, 14.0f,   15.0f, 16.0f,
         1.0f,  2.0f,    3.0f,  4.0f,    5.0f,  6.0f,    7.0f,  8.0f,
         9.0f, 10.0f,   11.0f, 12.0f,   13.0f, 14.0f,   15.0f, 16.0f,

        -2.0f,  0.0f,   19.0f, 20.0f,   21.0f, -1.0f,   15.0f, -5.0f,
        21.0f, -1.0f,   15.0f, -5.0f,   19.0f, 20.0f,   21.0f, -1.0f,
        13.0f,  7.0f,    1.0f, 16.0f,   15.0f, -5.0f,   19.0f, 20.0f,
         8.0f, 14.0f,   15.0f, -3.0f,   15.0f, -3.0f,    7.0f,  1.0f
        });

    Tensor result = layer.forwardPropagation(tensor);

    ASSERT_EQ( 4, result.getDim());
    ASSERT_EQ( 2, result.getShape()[0]);
    ASSERT_EQ( 2, result.getShape()[1]);
    ASSERT_EQ( 2, result.getShape()[2]);
    ASSERT_EQ( 2, result.getShape()[3]);

    ASSERT_EQ( 11.0f, result.getValue({ 0, 0, 0, 0 }));
    ASSERT_EQ( 12.0f, result.getValue({ 0, 0, 0, 1 }));
    ASSERT_EQ( 15.0f, result.getValue({ 0, 0, 1, 0 }));
    ASSERT_EQ( 16.0f, result.getValue({ 0, 0, 1, 1 }));
    ASSERT_EQ( 11.0f, result.getValue({ 0, 1, 0, 0 }));
    ASSERT_EQ( 12.0f, result.getValue({ 0, 1, 0, 1 }));
    ASSERT_EQ( 15.0f, result.getValue({ 0, 1, 1, 0 }));
    ASSERT_EQ( 16.0f, result.getValue({ 0, 1, 1, 1 }));
    ASSERT_EQ( 21.0f, result.getValue({ 1, 0, 0, 0 }));
    ASSERT_EQ( 20.0f, result.getValue({ 1, 0, 0, 1 }));
    ASSERT_EQ( 21.0f, result.getValue({ 1, 0, 1, 0 }));
    ASSERT_EQ( 20.0f, result.getValue({ 1, 0, 1, 1 }));
    ASSERT_EQ( 15.0f, result.getValue({ 1, 1, 0, 0 }));
    ASSERT_EQ( 16.0f, result.getValue({ 1, 1, 0, 1 }));
    ASSERT_EQ( 19.0f, result.getValue({ 1, 1, 1, 0 }));
    ASSERT_EQ( 20.0f, result.getValue({ 1, 1, 1, 1 }));
}
