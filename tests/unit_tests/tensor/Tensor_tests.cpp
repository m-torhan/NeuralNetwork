#include <gtest/gtest.h>
#include "src/Tensor.h"

TEST(Tensor_test, WhenGetValueShouldReturnProperItem) {
    Tensor tensor = Tensor({ 3, 3 });

    tensor[{ 1, 2 }] = .5f;

    ASSERT_EQ(.5f, (const_cast<const Tensor&>(tensor)[{ 1, 2 }]));
}

TEST(Tensor_test, WhenGetValueZeroDimensionalTensorShouldReturnNumber) {
    Tensor tensor = Tensor();

    tensor[{ 0 }] = 1.0f;

    ASSERT_EQ(1.0f, (const_cast<const Tensor&>(tensor)[{ 0 }]));
}

TEST(Tensor_test, WhenAddZeroDimensionalTensorsShouldBehaveLikeNumbers) {
    Tensor tensor_a = Tensor();
    Tensor tensor_b = Tensor();

    tensor_a[{ 0 }] = 6.0f;
    tensor_b[{ 0 }] = 2.0f;

    tensor_a += tensor_b;

    ASSERT_EQ(8.0f, (const_cast<const Tensor&>(tensor_a)[{ 0 }]));
}

TEST(Tensor_test, WhenMultiplyZeroDimensionalTensorsShouldBehaveLikeNumbers) {
    Tensor tensor_a = Tensor();
    Tensor tensor_b = Tensor();

    tensor_a[{ 0 }] = 0.5f;
    tensor_b[{ 0 }] = 4.0f;

    tensor_a *= tensor_b;

    ASSERT_EQ(2.0f, (const_cast<const Tensor&>(tensor_a)[{ 0 }]));
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

TEST(Tensor_test, GetSubTensorTestOneAxis) {
    Tensor tensor = Tensor({ 2, 3, 4 });

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 23.0f
    });

    const Tensor sub_tensor = const_cast<const Tensor&>(tensor)[{ {{ 0 }}, {}, {{ 2 }} }];
    
    ASSERT_EQ( 1u, sub_tensor.getDim());
    ASSERT_EQ( 3u, sub_tensor.getShape()[0]);

    ASSERT_EQ( 3.0f, sub_tensor[{ 0 }]);
    ASSERT_EQ( 7.0f, sub_tensor[{ 1 }]);
    ASSERT_EQ(11.0f, sub_tensor[{ 2 }]);
}

TEST(Tensor_test, GetSubTensorTestTwoAxes) {
    Tensor tensor = Tensor({ 2, 3, 4 });

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 23.0f
    });

    const Tensor sub_tensor = const_cast<const Tensor&>(tensor)[{ {}, {{ 1 }}, {} }];
    
    ASSERT_EQ( 2u, sub_tensor.getDim());
    ASSERT_EQ( 2u, sub_tensor.getShape()[0]);
    ASSERT_EQ( 4u, sub_tensor.getShape()[1]);

    ASSERT_EQ( 5.0f, (sub_tensor[{ 0, 0 }]));
    ASSERT_EQ( 6.0f, (sub_tensor[{ 0, 1 }]));
    ASSERT_EQ( 7.0f, (sub_tensor[{ 0, 2 }]));
    ASSERT_EQ( 8.0f, (sub_tensor[{ 0, 3 }]));
    ASSERT_EQ(17.0f, (sub_tensor[{ 1, 0 }]));
    ASSERT_EQ(18.0f, (sub_tensor[{ 1, 1 }]));
    ASSERT_EQ(19.0f, (sub_tensor[{ 1, 2 }]));
    ASSERT_EQ(20.0f, (sub_tensor[{ 1, 3 }]));
}

TEST(Tensor_test, GetSubTensorTestOneAxisWithRanges) {
    Tensor tensor = Tensor({ 2, 3, 4 });

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 23.0f
    });

    const Tensor sub_tensor = const_cast<const Tensor&>(tensor)[{ {{ 0 }}, {}, {{ 2 }} }];
    
    ASSERT_EQ( 1u, sub_tensor.getDim());
    ASSERT_EQ( 3u, sub_tensor.getShape()[0]);

    ASSERT_EQ( 3.0f, sub_tensor[{ 0 }]);
    ASSERT_EQ( 7.0f, sub_tensor[{ 1 }]);
    ASSERT_EQ(11.0f, sub_tensor[{ 2 }]);
}

TEST(Tensor_test, GetSubTensorTestTwoAxesWithRanges) {
    Tensor tensor = Tensor({ 2, 3, 4 });

    tensor.setValues({
         1.0f,  2.0f,  3.0f,  4.0f,
         5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f,

        13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f,
        21.0f, 22.0f, 23.0f, 23.0f
    });

    const Tensor sub_tensor = const_cast<const Tensor&>(tensor)[{ { 0 }, {}, {1, 3} }];
    
    ASSERT_EQ( 2u, sub_tensor.getDim());
    ASSERT_EQ( 3u, sub_tensor.getShape()[0]);
    ASSERT_EQ( 2u, sub_tensor.getShape()[1]);

    ASSERT_EQ( 2.0f, (sub_tensor[{ 0, 0 }]));
    ASSERT_EQ( 3.0f, (sub_tensor[{ 0, 1 }]));
    ASSERT_EQ( 6.0f, (sub_tensor[{ 1, 0 }]));
    ASSERT_EQ( 7.0f, (sub_tensor[{ 1, 1 }]));
    ASSERT_EQ(10.0f, (sub_tensor[{ 2, 0 }]));
    ASSERT_EQ(11.0f, (sub_tensor[{ 2, 1 }]));
}

TEST(Tensor_test, SetValuesOfSubTensorTestOneAxis) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 3 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,
    });

    tensor[{ {{ 0 }}, {}, {{ 2 }} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 2 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 2 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxes) {
    Tensor tensor = Tensor({ 2, 3, 1 });
    Tensor sub_tensor = Tensor({ 2 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f
    });

    tensor[{ {{ 0 }}, {{ 0, 2 }}, {{ 0 }} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxesFirstAxis) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 3, 4 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f,
        9.0f, 10.0f, 11.0f, 12.0f
    });

    tensor[std::vector<std::vector<uint32_t>>({ {{ 0 }}, {}, {} })] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 1 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 2 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0, 3 }]));
    ASSERT_EQ( 5.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));
    ASSERT_EQ( 6.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 1 }]));
    ASSERT_EQ( 7.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 8.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 3 }]));
    ASSERT_EQ( 9.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 0 }]));
    ASSERT_EQ(10.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 1 }]));
    ASSERT_EQ(11.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 2 }]));
    ASSERT_EQ(12.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 3 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxesSecondAxis) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 2, 4 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f
    });

    tensor[{ {}, {{ 1 }}, {} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 1 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 3 }]));
    ASSERT_EQ( 5.0f, (const_cast<const Tensor&>(tensor)[{ 1, 1, 0 }]));
    ASSERT_EQ( 6.0f, (const_cast<const Tensor&>(tensor)[{ 1, 1, 1 }]));
    ASSERT_EQ( 7.0f, (const_cast<const Tensor&>(tensor)[{ 1, 1, 2 }]));
    ASSERT_EQ( 8.0f, (const_cast<const Tensor&>(tensor)[{ 1, 1, 3 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxesFirstAxisWithRanges) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 1, 2, 3 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,
        4.0f,  5.0f,  6.0f,
    });

    tensor[{ {0, 1}, {1, 3}, {0, 3} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 1 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 0 }]));
    ASSERT_EQ( 5.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 1 }]));
    ASSERT_EQ( 6.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 2 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxesFirstAxisWithRangesWithWholeAxis) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 1, 2, 4 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f
    });

    tensor[{ { 0, 1 }, { 1, 3 }, {} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 1 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 3 }]));
    ASSERT_EQ( 5.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 0 }]));
    ASSERT_EQ( 6.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 1 }]));
    ASSERT_EQ( 7.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 2 }]));
    ASSERT_EQ( 8.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 3 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, SetValuesOfSubTensorTestTwoAxesFirstAxisWithRangesWithSingleIndex) {
    Tensor tensor = Tensor({ 2, 3, 4 });
    Tensor sub_tensor = Tensor({ 2, 4 });

    tensor *= 0.0f;

    sub_tensor.setValues({
        1.0f,  2.0f,  3.0f,  4.0f,
        5.0f,  6.0f,  7.0f,  8.0f
    });

    tensor[{ { 0 }, { 1, 3 }, {} }] = sub_tensor;

    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 0 }]));
    ASSERT_EQ( 2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 1 }]));
    ASSERT_EQ( 3.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 2 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1, 3 }]));
    ASSERT_EQ( 5.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 0 }]));
    ASSERT_EQ( 6.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 1 }]));
    ASSERT_EQ( 7.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 2 }]));
    ASSERT_EQ( 8.0f, (const_cast<const Tensor&>(tensor)[{ 0, 2, 3 }]));

    ASSERT_EQ(sub_tensor.sum(), tensor.sum());
}

TEST(Tensor_test, AddPaddingOneAxisLeftTest) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });
    
    const Tensor result = tensor.addPadding({ 0 }, { Left }, { 2 });

    ASSERT_EQ(4u, result.getShape()[0]);
    ASSERT_EQ(3u, result.getShape()[1]);
    ASSERT_EQ(2u, result.getShape()[2]);

    ASSERT_EQ(result.sum(), tensor.sum());
}

TEST(Tensor_test, AddPaddingTwoAxesTest) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });
    
    const Tensor result = tensor.addPadding({ 0, 2 }, { Left, Both }, { 2, 3 });

    ASSERT_EQ(4u, result.getShape()[0]);
    ASSERT_EQ(3u, result.getShape()[1]);
    ASSERT_EQ(8u, result.getShape()[2]);

    ASSERT_EQ(result.sum(), tensor.sum());
}

TEST(Tensor_test, WhenTensorPreceededByMinusEachValueShouldChangeSign) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, .5f,
        .25f, -2.0f
        });

    const Tensor result = -tensor;

    ASSERT_EQ(-1.0f, (result[{ 0, 0 }]));
    ASSERT_EQ( -.5f, (result[{ 0, 1 }]));
    ASSERT_EQ(-.25f, (result[{ 1, 0 }]));
    ASSERT_EQ( 2.0f, (result[{ 1, 1 }]));
}

TEST(Tensor_test, WhenAddedTwoTensorsEachValuePairShouldBeAdded) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 2, 3 });

    tensor_a.setValues({
        1.0f, .5f,   1.0f,
        .25f, .125f, 8.0f
        });

    tensor_b.setValues({
        16.0f, 8.0f, 2.0f,
        4.0f, 2.0f,  .5f
        });

    const Tensor tensor_c = tensor_a + tensor_b;

    ASSERT_EQ( 17.0f, (tensor_c[{ 0, 0 }]));
    ASSERT_EQ(  8.5f, (tensor_c[{ 0, 1 }]));
    ASSERT_EQ(  3.0f, (tensor_c[{ 0, 2 }]));
    ASSERT_EQ( 4.25f, (tensor_c[{ 1, 0 }]));
    ASSERT_EQ(2.125f, (tensor_c[{ 1, 1 }]));
    ASSERT_EQ(  8.5f, (tensor_c[{ 1, 2 }]));
}

TEST(Tensor_test, WhenAddedTwoTensorsWithDifferentShapesSecondTensorShoudBeAddedToEachRow) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 3 });

    tensor_a.setValues({
        1.0f, .5f,   1.0f,
        .25f, .125f, 8.0f
        });

    tensor_b.setValues({
        16.0f, 8.0f, 2.0f,
        });

    const Tensor tensor_c = tensor_a + tensor_b;

    ASSERT_EQ(2, (int)tensor_c.getDim());
    
    ASSERT_EQ(2, (int)tensor_c.getShape()[0]);
    ASSERT_EQ(3, (int)tensor_c.getShape()[1]);

    ASSERT_EQ( 17.0f, (tensor_c[{ 0, 0 }]));
    ASSERT_EQ(  8.5f, (tensor_c[{ 0, 1 }]));
    ASSERT_EQ(  3.0f, (tensor_c[{ 0, 2 }]));
    ASSERT_EQ(16.25f, (tensor_c[{ 1, 0 }]));
    ASSERT_EQ(8.125f, (tensor_c[{ 1, 1 }]));
    ASSERT_EQ( 10.0f, (tensor_c[{ 1, 2 }]));
}

TEST(Tensor_test, WhenTensorSubtractedFromNumberShouldBeReturnedTensorWithDifferences) {
    float number = 1.0f;
    Tensor tensor_a = Tensor({ 2, 3 });

    tensor_a.setValues({
        1.0f, .5f,   1.0f,
        .25f, .125f, 8.0f
        });

    const Tensor tensor_c = number - tensor_a;

    ASSERT_EQ(  0.0f, (tensor_c[{ 0, 0 }]));
    ASSERT_EQ(  0.5f, (tensor_c[{ 0, 1 }]));
    ASSERT_EQ( 0.00f, (tensor_c[{ 0, 2 }]));
    ASSERT_EQ( 0.75f, (tensor_c[{ 1, 0 }]));
    ASSERT_EQ(0.875f, (tensor_c[{ 1, 1 }]));
    ASSERT_EQ( -7.0f, (tensor_c[{ 1, 2 }]));
}

TEST(Tensor_test, WhenMultipliedTwoTensorsEachValuePairShouldBeMultiplied) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 2, 3 });

    tensor_a.setValues({
        1.0f, .5f,   1.0f,
        .25f, .125f, 8.0f
        });

    tensor_b.setValues({
        16.0f, 8.0f, 2.0f,
        4.0f, 2.0f,  .5f
        });

    const Tensor tensor_c = tensor_a * tensor_b;

    ASSERT_EQ(16.0f, (tensor_c[{ 0, 0 }]));
    ASSERT_EQ( 4.0f, (tensor_c[{ 0, 1 }]));
    ASSERT_EQ( 2.0f, (tensor_c[{ 0, 2 }]));
    ASSERT_EQ( 1.0f, (tensor_c[{ 1, 0 }]));
    ASSERT_EQ( .25f, (tensor_c[{ 1, 1 }]));
    ASSERT_EQ( 4.0f, (tensor_c[{ 1, 2 }]));
}

TEST(Tensor_test, WhenMultipliedInPlaceByTensorEachValuePairShouldBeMultiplied) {
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

    ASSERT_EQ(16.0f, (const_cast<const Tensor&>(tensor_a)[{ 0, 0 }]));
    ASSERT_EQ( 4.0f, (const_cast<const Tensor&>(tensor_a)[{ 0, 1 }]));
    ASSERT_EQ( 1.0f, (const_cast<const Tensor&>(tensor_a)[{ 1, 0 }]));
    ASSERT_EQ( .25f, (const_cast<const Tensor&>(tensor_a)[{ 1, 1 }]));
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

    ASSERT_EQ(4.0f, (const_cast<const Tensor&>(tensor_a)[{ 0, 0 }]));
    ASSERT_EQ(1.0f, (const_cast<const Tensor&>(tensor_a)[{ 0, 1 }]));
    ASSERT_EQ(1.0f, (const_cast<const Tensor&>(tensor_a)[{ 1, 0 }]));
    ASSERT_EQ(.25f, (const_cast<const Tensor&>(tensor_a)[{ 1, 1 }]));
}

TEST(Tensor_test, WhenMultipliedByNumberEachValueShouldBeMultiplied) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, .5f,
        .25f, .125f,
        });

    tensor *= 2;

    ASSERT_EQ(2.0f, (const_cast<const Tensor&>(tensor)[{ 0, 0 }]));
    ASSERT_EQ(1.0f, (const_cast<const Tensor&>(tensor)[{ 0, 1 }]));
    ASSERT_EQ( .5f, (const_cast<const Tensor&>(tensor)[{ 1, 0 }]));
    ASSERT_EQ(.25f, (const_cast<const Tensor&>(tensor)[{ 1, 1 }]));
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

    const Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(36.0f, result[{ 0 }]);
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

    const Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(18.0f, result[{ 0 }]);
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

    const Tensor result = tensor_a.dotProduct(tensor_b);

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(18.0f, result[{ 0 }]);
    ASSERT_EQ( 9.0f, result[{ 1 }]);
}

TEST(Tensor_test, WhenTensorsAreMatricesDotProductTransposeShouldReturnMatricesProductWhereSecondMatrixIsTransposed) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 2, 3 });

    tensor_a.setValues({
        1.0f, .5f,   2.0f,
        .25f, .125f, 1.0f
        });

    tensor_b.setValues({
        16.0f, 8.0f, 4.0f,
        4.0f,  2.0f, 2.0f
        });

    const Tensor result = tensor_a.dotProductTranspose(tensor_b);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(28.0f, (result[{ 0, 0 }]));
    ASSERT_EQ( 9.0f, (result[{ 0, 1 }]));
    ASSERT_EQ( 9.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(3.25f, (result[{ 1, 1 }]));
}

TEST(Tensor_test, TensorProductResultDimShouldBeSumOfArgumentsDims) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 4, 5, 6 });

    const Tensor result = tensor_a.tensorProduct(tensor_b);

    ASSERT_EQ(5, (int)result.getDim());
}

TEST(Tensor_test, TensorProductResultShapeShouldBeConcatOfArgumentsShapes) {
    Tensor tensor_a = Tensor({ 2, 3 });
    Tensor tensor_b = Tensor({ 4, 5, 6 });

    const Tensor result = tensor_a.tensorProduct(tensor_b);

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

    tensor_b[{ 0, 0, 0 }] =   .5f;
    tensor_b[{ 1, 2, 3 }] =  .25f;
    tensor_b[{ 1, 2, 0 }] = .125f;

    const Tensor result = tensor_a.tensorProduct(tensor_b);

    ASSERT_EQ( 1.0f, (result[{ 0, 0, 0, 0, 0 }]));
    ASSERT_EQ(32.0f, (result[{ 1, 2, 0, 0, 0 }]));
    ASSERT_EQ( 1.0f, (result[{ 0, 1, 1, 2, 3 }]));
    ASSERT_EQ( 8.0f, (result[{ 1, 1, 1, 2, 3 }]));
    ASSERT_EQ( 1.0f, (result[{ 0, 2, 1, 2, 0 }]));
    ASSERT_EQ( 2.0f, (result[{ 1, 0, 1, 2, 0 }]));
}

TEST(Tensor_test, ApplyFunctionShouldApplyGivenFunctionToTensor) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    const Tensor result = tensor.applyFunction([](float value) {return value * 2.0f; });

    ASSERT_EQ(2.0f, (result[{ 0, 0 }]));
    ASSERT_EQ(4.0f, (result[{ 0, 1 }]));
    ASSERT_EQ(6.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(8.0f, (result[{ 1, 1 }]));
}

TEST(Tensor_test, TensorSumTest) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    float result = tensor.sum();

    ASSERT_EQ(78.0f, result);
}

TEST(Tensor_test, SumAcrossFirstAxisOf2DTensor) {
    Tensor tensor = Tensor({ 4, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,
        7.0f, 8.0f
        });

    const Tensor result = tensor.sum(0);

    ASSERT_EQ(1, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);

    ASSERT_EQ(16.0f, result[{ 0 }]);
    ASSERT_EQ(20.0f, result[{ 1 }]);
}

TEST(Tensor_test, SumAcrossSecondAxisOf2DTensor) {
    Tensor tensor = Tensor({ 2, 3 });

    tensor.setValues({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
        });

    const Tensor result = tensor.sum(1);

    ASSERT_EQ(1, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);

    ASSERT_EQ( 6.0f, result[{ 0 }]);
    ASSERT_EQ(15.0f, result[{ 1 }]);
}

TEST(Tensor_test, SumAcrossFirstAxisOf3DTensor) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    const Tensor result = tensor.sum(0);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ(3, (int)result.getShape()[0]);
    ASSERT_EQ(2, (int)result.getShape()[1]);

    ASSERT_EQ( 8.0f, (result[{ 0, 0 }]));
    ASSERT_EQ(10.0f, (result[{ 0, 1 }]));
    ASSERT_EQ(12.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(14.0f, (result[{ 1, 1 }]));
    ASSERT_EQ(16.0f, (result[{ 2, 0 }]));
    ASSERT_EQ(18.0f, (result[{ 2, 1 }]));
}

TEST(Tensor_test, SumAcrossSecondAxisOf3DTensor) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    const Tensor result = tensor.sum(1);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(2, (int)result.getShape()[1]);

    ASSERT_EQ( 9.0f, (result[{ 0, 0 }]));
    ASSERT_EQ(12.0f, (result[{ 0, 1 }]));
    ASSERT_EQ(27.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(30.0f, (result[{ 1, 1 }]));
}

TEST(Tensor_test, SumAcrossThridAxisOf3DTensor) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f,
        5.0f, 6.0f,

        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
        });

    const Tensor result = tensor.sum(2);

    ASSERT_EQ(2, (int)result.getDim());

    ASSERT_EQ(2, (int)result.getShape()[0]);
    ASSERT_EQ(3, (int)result.getShape()[1]);

    ASSERT_EQ( 3.0f, (result[{ 0, 0 }]));
    ASSERT_EQ( 7.0f, (result[{ 0, 1 }]));
    ASSERT_EQ(11.0f, (result[{ 0, 2 }]));
    ASSERT_EQ(15.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(19.0f, (result[{ 1, 1 }]));
    ASSERT_EQ(23.0f, (result[{ 1, 2 }]));
}

TEST(Tensor_test, WhenFlattenResultShouldHaveOnlyOneDimEqualToSize) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    const Tensor result = tensor.flatten();

    ASSERT_EQ(1, (int)result.getDim());
    ASSERT_EQ(tensor.getSize(), result.getShape()[0]);
}

TEST(Tensor_test, WhenFlattenFromFirstAxisResultShouldBeTwoDimensionalAndFirstShapeShouldRemain) {
    Tensor tensor = Tensor({ 2, 3, 2 });

    const Tensor result = tensor.flatten(1);

    ASSERT_EQ(2, (int)result.getDim());
    ASSERT_EQ(tensor.getShape()[0], result.getShape()[0]);
    ASSERT_EQ(tensor.getSize()/tensor.getShape()[0], result.getShape()[1]);
}

TEST(Tensor_test, TensorShuffleShouldRearrangeValues) {
    Tensor tensor = Tensor({ 2, 2 });

    tensor.setValues({
        1.0f, 2.0f,
        3.0f, 4.0f
        });

    const Tensor result = tensor.shuffle();

    ASSERT_TRUE((result[{ 0, 0 }] == 1) || (result[{ 1, 0 }] == 1));
    ASSERT_TRUE((result[{ 0, 0 }] == 3) || (result[{ 1, 0 }] == 3));
    ASSERT_TRUE((result[{ 0, 1 }] == 2) || (result[{ 1, 1 }] == 2));
    ASSERT_TRUE((result[{ 0, 1 }] == 4) || (result[{ 1, 1 }] == 4));
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

    const Tensor result = tensor.shuffle(pattern);
    
    ASSERT_EQ(5.0f, (result[{ 0, 0 }]));
    ASSERT_EQ(7.0f, (result[{ 1, 0 }]));
    ASSERT_EQ(1.0f, (result[{ 2, 0 }]));
    ASSERT_EQ(3.0f, (result[{ 3, 0 }]));
}