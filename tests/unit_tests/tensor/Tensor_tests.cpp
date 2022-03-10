#include <gtest/gtest.h>
#include "src/Tensor.h"

TEST(Tensor_test, WhenGetValueShouldReturnProperItem) {
    Tensor tensor = Tensor({ 3, 3 });

    tensor.setValue(.5f, { 1, 2 });

    ASSERT_EQ(.5f, tensor.getValue({ 1, 2 }));
}

TEST(Tensor_test, WhenGetValueZeroDimensionalTensorShouldReturnNumber) {
    Tensor tensor = Tensor();

    tensor.setValue(1.0f);

    ASSERT_EQ(1.0f, tensor.getValue());
}

TEST(Tensor_test, WhenAddZeroDimensionalTensorsShouldBehaveLikeNumbers) {
    Tensor tensor_a = Tensor();
    Tensor tensor_b = Tensor();

    tensor_a.setValue(6.0f);
    tensor_b.setValue(2.0f);

    tensor_a += tensor_b;

    ASSERT_EQ(8.0f, tensor_a.getValue());
}

TEST(Tensor_test, WhenMultiplyZeroDimensionalTensorsShouldBehaveLikeNumbers) {
    Tensor tensor_a = Tensor();
    Tensor tensor_b = Tensor();

    tensor_a.setValue(0.5f);
    tensor_b.setValue(4.0f);

    tensor_a *= tensor_b;

    ASSERT_EQ(2.0f, tensor_a.getValue());
}
		
TEST(Tensor_test, SetValueShouldBeProperlyPlacedInData) {
    Tensor tensor = Tensor({ 2, 2, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,

        5.0f, 6.0f,
        7.0f, 8.0f
        });

    ASSERT_EQ(1.0f, tensor.getData()[0]);
    ASSERT_EQ(2.0f, tensor.getData()[1]);
    ASSERT_EQ(3.0f, tensor.getData()[2]);
    ASSERT_EQ(4.0f, tensor.getData()[3]);
    ASSERT_EQ(5.0f, tensor.getData()[4]);
    ASSERT_EQ(6.0f, tensor.getData()[5]);
    ASSERT_EQ(7.0f, tensor.getData()[6]);
    ASSERT_EQ(8.0f, tensor.getData()[7]);
}

TEST(Tensor_test, WhenTensorPreceededByMinusEachValueShouldChangeSign) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, .5f,
        .25f, -2.0f
        });

    Tensor result = -tensor;

    ASSERT_EQ(-1.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ( -.5f, result.getValue({ 0, 1 }));
    ASSERT_EQ(-.25f, result.getValue({ 1, 0 }));
    ASSERT_EQ( 2.0f, result.getValue({ 1, 1 }));
}

TEST(Tensor_test, WhenMultipliedByTensorEachValuePairShouldBeMultiplied) {
    Tensor tensor_a = Tensor({ 2, 2 });
    Tensor tensor_b = Tensor({ 2, 2 });

    tensor_a.setValues({
        1.0f, .5f,
        .25f, .125f
        });

    tensor_b.setValues({
        16.0f, 8.0f,
        4.0f, 2.0f
        });

    tensor_a *= tensor_b;

    ASSERT_EQ(16.0f, tensor_a.getValue({ 0, 0 }));
    ASSERT_EQ( 4.0f, tensor_a.getValue({ 0, 1 }));
    ASSERT_EQ( 1.0f, tensor_a.getValue({ 1, 0 }));
    ASSERT_EQ( .25f, tensor_a.getValue({ 1, 1 }));
}

TEST(Tensor_test, WhenMultipliedByRowTensorEachValuePairShouldBeMultiplied) {
    Tensor tensor_a = Tensor({ 2, 2 });
    Tensor tensor_b = Tensor({ 2 });

    tensor_a.setValues({
        1.0f, .5f,
        .25f, .125f
        });

    tensor_b.setValues({
        4.0f, 2.0f,
        });

    tensor_a *= tensor_b;

    ASSERT_EQ(4.0f, tensor_a.getValue({ 0, 0 }));
    ASSERT_EQ(1.0f, tensor_a.getValue({ 0, 1 }));
    ASSERT_EQ(1.0f, tensor_a.getValue({ 1, 0 }));
    ASSERT_EQ(.25f, tensor_a.getValue({ 1, 1 }));
}

TEST(Tensor_test, WhenMultipliedByNumberEachValueShouldBeMultiplied) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, .5f,
        .25f, .125f,
        });

    tensor *= 2;

    ASSERT_EQ(2.0f, tensor.getValue({ 0, 0 }));
    ASSERT_EQ(1.0f, tensor.getValue({ 0, 1 }));
    ASSERT_EQ( .5f, tensor.getValue({ 1, 0 }));
    ASSERT_EQ(.25f, tensor.getValue({ 1, 1 }));
}

TEST(Tensor_test, WhenTensorsAreTwoVectorsDotProductShouldReturnSumOfProductsOfAllPairs) {
    Tensor tensor_a = Tensor({ 8 });
    Tensor tensor_b = Tensor({ 8 });

    tensor_a.setValues({
        2.0f, .5f, 3.0f, 4.0f, 2.0f, 0.5f, 0.25f, 5.0f
        });

    tensor_b.setValues({
        8.0f, 4.0f, 1.0f, 0.25f, 0.5f, 2.0f, 8.0f, 2.0f
        });

    Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(36.0f, result.getValue());
}

TEST(Tensor_test, WhenTensorsAreTwoVectorsWithSizeNotAlignedDotProductShouldReturnSumOfProductsOfAllPairs) {
    Tensor tensor_a = Tensor({ 2 });
    Tensor tensor_b = Tensor({ 2 });

    tensor_a.setValues({
        2.0f, .5f,
        });

    tensor_b.setValues({
        8.0f, 4.0f
        });

    Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(18.0f, result.getValue());
}

TEST(Tensor_test, WhenTensorsAreMatrixAndVectorDotProductShouldReturnSumsOfProductsOfRowsByVector) {
    Tensor tensor_a = Tensor({ 2, 2 });
    Tensor tensor_b = Tensor({ 2 });

    tensor_a.setValues({
        1.0f, .5f,
        .25f, .125f
        });

    tensor_b.setValues({
        16.0f, 8.0f
        });

    Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(18.0f, result.getValue({ 0 }));
    ASSERT_EQ( 9.0f, result.getValue({ 1 }));
}

TEST(Tensor_test, TensorProductResultDimShouldBeSumOfArgumentsDims) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 4, 5, 6 });

    Tensor result = tensor_a.tensorProduct(tensor_b);

    ASSERT_EQ(5, (int)result.getDim());
}

TEST(Tensor_test, TensorProductResultShapeShouldBeConcatOfArgumentsShapes) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 4, 5, 6 });

    Tensor result = tensor_a.tensorProduct(tensor_b);

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);
    ASSERT_EQ(4, (int)result.getShape()[2]);
    ASSERT_EQ(5, (int)result.getShape()[3]);
    ASSERT_EQ(6, (int)result.getShape()[4]);
}

TEST(Tensor_test, TensorProductResultShouldBeCorrect) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 4, 5, 6 });

    tensor_a.setValues({
        2.0f, 4.0f, 8.0f,
        16.0f, 32.0f, 64.0f
        });

    tensor_b.setValue(  .5f, { 0, 0, 0 });
    tensor_b.setValue( .25f, { 1, 2, 3 });
    tensor_b.setValue(.125f, { 1, 2, 0 });

    Tensor result = tensor_a.tensorProduct(tensor_b);

    ASSERT_EQ( 1.0f, result.getValue({ 0, 0, 0, 0, 0 }));
    ASSERT_EQ(32.0f, result.getValue({ 1, 2, 0, 0, 0 }));
    ASSERT_EQ( 1.0f, result.getValue({ 0, 1, 1, 2, 3 }));
    ASSERT_EQ( 8.0f, result.getValue({ 1, 1, 1, 2, 3 }));
    ASSERT_EQ( 1.0f, result.getValue({ 0, 2, 1, 2, 0 }));
    ASSERT_EQ( 2.0f, result.getValue({ 1, 0, 1, 2, 0 }));
}

TEST(Tensor_test, ApplyFunctionShouldApplyGivenFunctionToTensor) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    Tensor result = tensor.applyFunction([](float value) {return value * 2.0f; });

    ASSERT_EQ(2.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ(4.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ(6.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(8.0f, result.getValue({ 1, 1 }));
}

TEST(Tensor_test, SumAcrossFirstAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.sum(0);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ( 8.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ(10.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ(12.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(14.0f, result.getValue({ 1, 1 }));
    ASSERT_EQ(16.0f, result.getValue({ 2, 0 }));
    ASSERT_EQ(18.0f, result.getValue({ 2, 1 }));
}

TEST(Tensor_test, SumAcrossSecondAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.sum(1);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ( 9.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ(12.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ(27.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(30.0f, result.getValue({ 1, 1 }));
}

TEST(Tensor_test, SumAcrossThridAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.sum(2);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ( 3.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ( 7.0f, result.getValue({ 0, 1 }));
    ASSERT_EQ(11.0f, result.getValue({ 0, 2 }));
    ASSERT_EQ(15.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(19.0f, result.getValue({ 1, 1 }));
    ASSERT_EQ(23.0f, result.getValue({ 1, 2 }));
}

TEST(Tensor_test, WhenFlattenResultShouldHaveOnlyOneDimEqualToSize) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    Tensor result = tensor.flatten();

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(tensor.getSize(), result.getShape()[0]);
}

TEST(Tensor_test, WhenFlattenFromFirstAxisResultShouldBeTwoDimensionalAndFirstShapeShouldRemain) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    Tensor result = tensor.flatten(1);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(tensor.getShape()[0], result.getShape()[0]);
    ASSERT_EQ(tensor.getSize()/tensor.getShape()[0], result.getShape()[1]);
}

TEST(Tensor_test, TensorSliceFirstAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.slice(0, 1, 2);

    ASSERT_EQ(3, (int)result.getDim());

    ASSERT_EQ(1, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);
    ASSERT_EQ(2, (int)result.getShape()[2]);

    ASSERT_EQ( 7.0f, result.getValue({ 0, 0, 0 }));
    ASSERT_EQ( 8.0f, result.getValue({ 0, 0, 1 }));
    ASSERT_EQ( 9.0f, result.getValue({ 0, 1, 0 }));
    ASSERT_EQ(10.0f, result.getValue({ 0, 1, 1 }));
    ASSERT_EQ(11.0f, result.getValue({ 0, 2, 0 }));
    ASSERT_EQ(12.0f, result.getValue({ 0, 2, 1 }));
}

TEST(Tensor_test, TensorSliceSecondAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.slice(1, 1, 2);

    ASSERT_EQ(3, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(1, (int)result.getShape()[1]);
    ASSERT_EQ(2, (int)result.getShape()[2]);

    ASSERT_EQ( 3.0f, result.getValue({ 0, 0, 0 }));
    ASSERT_EQ( 4.0f, result.getValue({ 0, 0, 1 }));
    ASSERT_EQ( 9.0f, result.getValue({ 1, 0, 0 }));
    ASSERT_EQ(10.0f, result.getValue({ 1, 0, 1 }));
}

TEST(Tensor_test, TensorSliceThirdAxis) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    Tensor result = tensor.slice(2, 1, 2);

    ASSERT_EQ(3, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);
    ASSERT_EQ(1, (int)result.getShape()[2]);

    ASSERT_EQ( 2.0f, result.getValue({ 0, 0, 0 }));
    ASSERT_EQ( 4.0f, result.getValue({ 0, 1, 0 }));
    ASSERT_EQ( 6.0f, result.getValue({ 0, 2, 0 }));
    ASSERT_EQ( 8.0f, result.getValue({ 1, 0, 0 }));
    ASSERT_EQ(10.0f, result.getValue({ 1, 1, 0 }));
    ASSERT_EQ(12.0f, result.getValue({ 1, 2, 0 }));
}

TEST(Tensor_test, TensorShuffleShouldRearrangeValues) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    Tensor result = tensor.shuffle();

    ASSERT_TRUE(result.getValue({ 0, 0 }) == 1 || result.getValue({ 1, 0 }) == 1);
    ASSERT_TRUE(result.getValue({ 0, 0 }) == 3 || result.getValue({ 1, 0 }) == 3);
    ASSERT_TRUE(result.getValue({ 0, 1 }) == 2 || result.getValue({ 1, 1 }) == 2);
    ASSERT_TRUE(result.getValue({ 0, 1 }) == 4 || result.getValue({ 1, 1 }) == 4);
}

TEST(Tensor_test, TensorShuffleWithPatternShouldRearrangeValues) {
    Tensor tensor = Tensor({ 4, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
        });

    uint32_t pattern[4] = { 2, 3, 0, 1 };

    Tensor result = tensor.shuffle(pattern);
    
    ASSERT_EQ(5.0f, result.getValue({ 0, 0 }));
    ASSERT_EQ(7.0f, result.getValue({ 1, 0 }));
    ASSERT_EQ(1.0f, result.getValue({ 2, 0 }));
    ASSERT_EQ(3.0f, result.getValue({ 3, 0 }));
}