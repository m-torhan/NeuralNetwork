#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetwork/Tensor.cpp"
#include "../NeuralNetwork/Layer.cpp"
#include "../NeuralNetwork/ActivationLayer.cpp"
#include "../NeuralNetwork/DenseLayer.cpp"
#include "../NeuralNetwork/NeuralNetwork.cpp"
#include "../NeuralNetwork/Utils.cpp"


using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkUnitTests {
	TEST_CLASS(UtilsUnitTests) { 
	public:
		TEST_METHOD(GenPermutationResultShouldContainAllNumbersUpToN) {
			bool b[32] = { false };
			uint32_t* permutation;
			int s = 0;
			int i = 0;

			permutation = genPermutation(32);

			for (i = 0; i < 32; ++i) {
				b[permutation[i]] = true;
			}
			for (i = 0; i < 32; ++i) {
				s += b[i];
			}
			Assert::AreEqual(32, s);
		}
	};

	TEST_CLASS(TensorUnitTests) {
	public:
		
		TEST_METHOD(WhenGetValueShouldReturnProperItem) {
			Tensor tensor = Tensor({ 3, 3 });

			tensor.setValue(.5f, { 1, 2 });

			Assert::AreEqual(.5f, tensor.getValue({ 1, 2 }));
		}

		TEST_METHOD(WhenGetValueZeroDimensionalTensorShouldReturnNumber) {
			Tensor tensor = Tensor();

			tensor.setValue(1.0f);

			Assert::AreEqual(1.0f, tensor.getValue());
		}

		TEST_METHOD(WhenAddZeroDimensionalTensorsShouldBehaveLikeNumbers) {
			Tensor tensor_a = Tensor();
			Tensor tensor_b = Tensor();

			tensor_a.setValue(6.0f);
			tensor_b.setValue(2.0f);

			tensor_a += tensor_b;

			Assert::AreEqual(8.0f, tensor_a.getValue());
		}

		TEST_METHOD(WhenMultiplyZeroDimensionalTensorsShouldBehaveLikeNumbers) {
			Tensor tensor_a = Tensor();
			Tensor tensor_b = Tensor();

			tensor_a.setValue(0.5f);
			tensor_b.setValue(4.0f);

			tensor_a *= tensor_b;

			Assert::AreEqual(2.0f, tensor_a.getValue());
		}
		
		TEST_METHOD(SetValueShouldBeProperlyPlacedInData) {
			Tensor tensor = Tensor({ 2, 2, 2 });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f,

				5.0f, 6.0f,
				7.0f, 8.0f
				});

			Assert::AreEqual(1.0f, tensor.getData()[0]);
			Assert::AreEqual(2.0f, tensor.getData()[1]);
			Assert::AreEqual(3.0f, tensor.getData()[2]);
			Assert::AreEqual(4.0f, tensor.getData()[3]);
			Assert::AreEqual(5.0f, tensor.getData()[4]);
			Assert::AreEqual(6.0f, tensor.getData()[5]);
			Assert::AreEqual(7.0f, tensor.getData()[6]);
			Assert::AreEqual(8.0f, tensor.getData()[7]);
		}

		TEST_METHOD(WhenTensorPreceededByMinusEachValueShouldChangeSign) {
			Tensor tensor = Tensor({ 2, 2 });

			tensor.setValues({
				1.0f, .5f,
				.25f, -2.0f
				});

			Tensor result = -tensor;

			Assert::AreEqual(-1.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual( -.5f, result.getValue({ 0, 1 }));
			Assert::AreEqual(-.25f, result.getValue({ 1, 0 }));
			Assert::AreEqual( 2.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(WhenMultipliedByTensorEachValuePairShouldBeMultiplied) {
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

			Assert::AreEqual(16.0f, tensor_a.getValue({ 0, 0 }));
			Assert::AreEqual( 4.0f, tensor_a.getValue({ 0, 1 }));
			Assert::AreEqual( 1.0f, tensor_a.getValue({ 1, 0 }));
			Assert::AreEqual( .25f, tensor_a.getValue({ 1, 1 }));
		}

		TEST_METHOD(WhenMultipliedByRowTensorEachValuePairShouldBeMultiplied) {
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

			Assert::AreEqual(4.0f, tensor_a.getValue({ 0, 0 }));
			Assert::AreEqual(1.0f, tensor_a.getValue({ 0, 1 }));
			Assert::AreEqual(1.0f, tensor_a.getValue({ 1, 0 }));
			Assert::AreEqual(.25f, tensor_a.getValue({ 1, 1 }));
		}

		TEST_METHOD(WhenMultipliedByNumberEachValueShouldBeMultiplied) {
			Tensor tensor = Tensor({ 2, 2 });

			tensor.setValues({
				1.0f, .5f,
				.25f, .125f,
				});

			tensor *= 2;

			Assert::AreEqual(2.0f, tensor.getValue({ 0, 0 }));
			Assert::AreEqual(1.0f, tensor.getValue({ 0, 1 }));
			Assert::AreEqual( .5f, tensor.getValue({ 1, 0 }));
			Assert::AreEqual(.25f, tensor.getValue({ 1, 1 }));
		}

		TEST_METHOD(WhenTensorsAreMatrixAndVectorDotProductShouldReturnSumsOfProductsOfRowsByVector) {
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

			Assert::AreEqual(1, (int)result.getDim());
			Assert::AreEqual(18.0f, result.getValue({ 0 }));
			Assert::AreEqual( 9.0f, result.getValue({ 1 }));
		}

		TEST_METHOD(TensorProductResultDimShouldBeSumOfArgumentsDims) {
			Tensor tensor_a = Tensor({ 2, 3 });
			Tensor tensor_b = Tensor({ 4, 5, 6 });

			Tensor result = tensor_a.tensorProduct(tensor_b);

			Assert::AreEqual(5, (int)result.getDim());
		}

		TEST_METHOD(TensorProductResultShapeShouldBeConcatOfArgumentsShapes) {
			Tensor tensor_a = Tensor({ 2, 3 });
			Tensor tensor_b = Tensor({ 4, 5, 6 });

			Tensor result = tensor_a.tensorProduct(tensor_b);

			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(3, (int)result.getShape()[1]);
			Assert::AreEqual(4, (int)result.getShape()[2]);
			Assert::AreEqual(5, (int)result.getShape()[3]);
			Assert::AreEqual(6, (int)result.getShape()[4]);
		}

		TEST_METHOD(TensorProductResultShouldBeCorrect) {
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

			Assert::AreEqual( 1.0f, result.getValue({ 0, 0, 0, 0, 0 }));
			Assert::AreEqual(32.0f, result.getValue({ 1, 2, 0, 0, 0 }));
			Assert::AreEqual( 1.0f, result.getValue({ 0, 1, 1, 2, 3 }));
			Assert::AreEqual( 8.0f, result.getValue({ 1, 1, 1, 2, 3 }));
			Assert::AreEqual( 1.0f, result.getValue({ 0, 2, 1, 2, 0 }));
			Assert::AreEqual( 2.0f, result.getValue({ 1, 0, 1, 2, 0 }));
		}

		TEST_METHOD(ApplyFunctionShouldApplyGivenFunctionToTensor) {
			Tensor tensor = Tensor({ 2, 2 });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f
				});

			Tensor result = tensor.applyFunction([](float value) {return value * 2.0f; });

			Assert::AreEqual(2.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual(4.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(6.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(8.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(SumAcrossFirstAxis) {
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

			Assert::AreEqual(2, (int)result.getDim());

			Assert::AreEqual( 8.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual(10.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(12.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(14.0f, result.getValue({ 1, 1 }));
			Assert::AreEqual(16.0f, result.getValue({ 2, 0 }));
			Assert::AreEqual(18.0f, result.getValue({ 2, 1 }));
		}

		TEST_METHOD(SumAcrossSecondAxis) {
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

			Assert::AreEqual(2, (int)result.getDim());

			Assert::AreEqual( 9.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual(12.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(27.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(30.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(SumAcrossThridAxis) {
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

			Assert::AreEqual(2, (int)result.getDim());

			Assert::AreEqual( 3.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual( 7.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(11.0f, result.getValue({ 0, 2 }));
			Assert::AreEqual(15.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(19.0f, result.getValue({ 1, 1 }));
			Assert::AreEqual(23.0f, result.getValue({ 1, 2 }));
		}

		TEST_METHOD(WhenFlattenResultShouldHaveOnlyOneDimEqualToSize) {
			Tensor tensor = Tensor({ 2, 3, 2 });

			Tensor result = tensor.flatten();

			Assert::AreEqual(1, (int)result.getDim());
			Assert::AreEqual(tensor.getSize(), result.getShape()[0]);
		}

		TEST_METHOD(WhenFlattenFromFirstAxisResultShouldBeTwoDimensionalAndFirstShapeShouldRemain) {
			Tensor tensor = Tensor({ 2, 3, 2 });

			Tensor result = tensor.flatten(1);

			Assert::AreEqual(2, (int)result.getDim());
			Assert::AreEqual(tensor.getShape()[0], result.getShape()[0]);
			Assert::AreEqual(tensor.getSize()/tensor.getShape()[0], result.getShape()[1]);
		}

		TEST_METHOD(TensorSliceFirstAxis) {
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

			Assert::AreEqual(3, (int)result.getDim());

			Assert::AreEqual(1, (int)result.getShape()[0]);
			Assert::AreEqual(3, (int)result.getShape()[1]);
			Assert::AreEqual(2, (int)result.getShape()[2]);

			Assert::AreEqual( 7.0f, result.getValue({ 0, 0, 0 }));
			Assert::AreEqual( 8.0f, result.getValue({ 0, 0, 1 }));
			Assert::AreEqual( 9.0f, result.getValue({ 0, 1, 0 }));
			Assert::AreEqual(10.0f, result.getValue({ 0, 1, 1 }));
			Assert::AreEqual(11.0f, result.getValue({ 0, 2, 0 }));
			Assert::AreEqual(12.0f, result.getValue({ 0, 2, 1 }));
		}

		TEST_METHOD(TensorSliceSecondAxis) {
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

			Assert::AreEqual(3, (int)result.getDim());

			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(1, (int)result.getShape()[1]);
			Assert::AreEqual(2, (int)result.getShape()[2]);

			Assert::AreEqual( 3.0f, result.getValue({ 0, 0, 0 }));
			Assert::AreEqual( 4.0f, result.getValue({ 0, 0, 1 }));
			Assert::AreEqual( 9.0f, result.getValue({ 1, 0, 0 }));
			Assert::AreEqual(10.0f, result.getValue({ 1, 0, 1 }));
		}

		TEST_METHOD(TensorSliceThirdAxis) {
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

			Assert::AreEqual(3, (int)result.getDim());

			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(3, (int)result.getShape()[1]);
			Assert::AreEqual(1, (int)result.getShape()[2]);

			Assert::AreEqual( 2.0f, result.getValue({ 0, 0, 0 }));
			Assert::AreEqual( 4.0f, result.getValue({ 0, 1, 0 }));
			Assert::AreEqual( 6.0f, result.getValue({ 0, 2, 0 }));
			Assert::AreEqual( 8.0f, result.getValue({ 1, 0, 0 }));
			Assert::AreEqual(10.0f, result.getValue({ 1, 1, 0 }));
			Assert::AreEqual(12.0f, result.getValue({ 1, 2, 0 }));
		}

		TEST_METHOD(TensorShuffleShouldRearrangeValues) {
			Tensor tensor = Tensor({ 2, 2 });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f
				});

			Tensor result = tensor.shuffle();

			Assert::IsTrue(result.getValue({ 0, 0 }) == 1 || result.getValue({ 1, 0 }) == 1);
			Assert::IsTrue(result.getValue({ 0, 0 }) == 3 || result.getValue({ 1, 0 }) == 3);
			Assert::IsTrue(result.getValue({ 0, 1 }) == 2 || result.getValue({ 1, 1 }) == 2);
			Assert::IsTrue(result.getValue({ 0, 1 }) == 4 || result.getValue({ 1, 1 }) == 4);
		}

		TEST_METHOD(TensorShuffleWithPatternShouldRearrangeValues) {
			Tensor tensor = Tensor({ 4, 2 });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f,
				5.0f, 6.0f,
				7.0f, 8.0f
				});

			uint32_t pattern[4] = { 2, 3, 0, 1 };

			Tensor result = tensor.shuffle(pattern);
			
			Assert::AreEqual(5.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual(7.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(1.0f, result.getValue({ 2, 0 }));
			Assert::AreEqual(3.0f, result.getValue({ 3, 0 }));
		}
	};

	TEST_CLASS(ActivationLayerUnitTests) {
	public:

		TEST_METHOD(ActivationLayerShouldApplyGivenFunctionWhenPrograpateForward) {
			Tensor tensor = Tensor({ 2, 2 });
			ActivationLayer layer = ActivationLayer({ 2, 2 }, [](const Tensor& x) -> const Tensor { return x * x; }, [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f
				});

			Tensor result = layer.forwardPropagation(tensor);

			Assert::AreEqual( 1.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual( 4.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual( 9.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(16.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(ActivationLayerShouldApplyGivenFunctionDerivativeWhenPrograpateBackward) {
			Tensor tensor = Tensor({ 2, 2 });
			ActivationLayer layer = ActivationLayer({ 2, 2 }, [](const Tensor& x) -> const Tensor { return x * x; }, [](const Tensor& x, const Tensor& dx) -> const Tensor { return x * (dx * 2.0f); });

			tensor.setValues({
				1.0f, 2.0f,
				3.0f, 4.0f
				});

			layer.forwardPropagation(tensor);
			Tensor result = layer.backwardPropagation(tensor);

			Assert::AreEqual( 2.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual( 8.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(18.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(32.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(SigmoidActivationValuesTest) {
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

			Assert::IsTrue(abs(0.26894f - forward.getValue({ 0 })) < 0.001f);
			Assert::IsTrue(abs(0.5f - forward.getValue({ 1 })) < 0.001f);
			Assert::IsTrue(abs(0.73106f - forward.getValue({ 2 })) < 0.001f);

			Assert::IsTrue(abs(0.04915f - backward.getValue({ 0 })) < 0.001f);
			Assert::IsTrue(abs(0.1f - backward.getValue({ 1 })) < 0.001f);
			Assert::IsTrue(abs(-0.29491f - backward.getValue({ 2 })) < 0.001f);
		}

		TEST_METHOD(ReLUActivationValuesTest) {
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

			Assert::IsTrue(abs(0.0f - forward.getValue({ 0 })) < 0.001f);
			Assert::IsTrue(abs(0.25f - forward.getValue({ 1 })) < 0.001f);
			Assert::IsTrue(abs(1.0f - forward.getValue({ 2 })) < 0.001f);

			Assert::IsTrue(abs(0.0f - backward.getValue({ 0 })) < 0.001f);
			Assert::IsTrue(abs(0.4f - backward.getValue({ 1 })) < 0.001f);
			Assert::IsTrue(abs(-1.5f - backward.getValue({ 2 })) < 0.001f);
		}
	};
	TEST_CLASS(DenseLayerUnitTests) {
	public:
		TEST_METHOD(DenseLayerForwardPropagationOutputShapeTest) {
			Tensor tensor = Tensor({ 2, 3, 4 });
			DenseLayer layer = DenseLayer({ 3, 4 }, 4);

			Tensor result = layer.forwardPropagation(tensor);

			Assert::AreEqual(2, (int)result.getDim());
			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(4, (int)result.getShape()[1]);
		}

		TEST_METHOD(DenseLayerBackwardPropagationOutputShapeTest) {
			Tensor tensor = Tensor({ 2, 12 });
			Tensor tensor_d = Tensor({ 2, 4 });
			DenseLayer layer = DenseLayer({ 3, 4 }, 4);

			layer.initCachedGradient();
			layer.forwardPropagation(tensor);
			Tensor result = layer.backwardPropagation(tensor_d);

			Assert::AreEqual(2, (int)result.getDim());
			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(12, (int)result.getShape()[1]);
		}

		TEST_METHOD(DenseLayerForwardPropagationReturnValuesTest) {
			Tensor tensor = Tensor({ 2, 3 });
			Tensor tensor_d = Tensor({ 2, 2 });
			DenseLayer layer = DenseLayer({ 3 }, 2);

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

			Tensor forward = layer.forwardPropagation(tensor);
			Tensor backward = layer.backwardPropagation(tensor_d);

			Assert::IsTrue(abs(0.6845f - forward.getValue({ 0, 0 })) < 0.001f);
			Assert::IsTrue(abs(-2.60345f - forward.getValue({ 0, 1 })) < 0.001f);
			Assert::IsTrue(abs(51.04f - forward.getValue({ 1, 0 })) < 0.001f);
			Assert::IsTrue(abs(-85.0583f - forward.getValue({ 1, 1 })) < 0.001f);

			Assert::IsTrue(abs(1.0145f - backward.getValue({ 0, 0 })) < 0.001f);
			Assert::IsTrue(abs(-0.48795f - backward.getValue({ 0, 1 })) < 0.001f);
			Assert::IsTrue(abs(0.551f - backward.getValue({ 0, 2 })) < 0.001f);
			Assert::IsTrue(abs(-23.7976f - backward.getValue({ 1, 0 })) < 0.001f);
			Assert::IsTrue(abs(10.71966f - backward.getValue({ 1, 1 })) < 0.001f);
			Assert::IsTrue(abs(-27.5218f - backward.getValue({ 1, 2 })) < 0.001f);
		}
	};

	TEST_CLASS(NeuralNetworkUnitTests) {
		TEST_METHOD(BinaryCrossentropyTest) {
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
			Tensor result_d = NeuralNetwork::binary_crossentropy_d(y_hat, y);

			Assert::IsTrue(abs(0.83766f - result) < 0.001f);

			Assert::AreEqual(2, (int)result_d.getDim());
			Assert::AreEqual(2, (int)result_d.getShape()[0]);
			Assert::AreEqual(2, (int)result_d.getShape()[1]);

			Assert::IsTrue(abs(-4.99999f - result_d.getValue({ 0, 0 })) < 0.001f);
			Assert::IsTrue(abs(0.0f - result_d.getValue({ 0, 1 })) < 0.001f);
			Assert::IsTrue(abs(1.03199f - result_d.getValue({ 1, 0 })) < 0.001f);
			Assert::IsTrue(abs(-0.59185f - result_d.getValue({ 1, 1 })) < 0.001f);
		}

		TEST_METHOD(PredictShouldReturnTensor) {
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

			Tensor result = nn.predict(tensor);

			Assert::AreEqual(2, (int)result.getDim());
			Assert::AreEqual(2, (int)result.getShape()[0]);
			Assert::AreEqual(2, (int)result.getShape()[1]);

			Assert::AreEqual(   1.0f, result.getValue({ 0, 0 }));
			Assert::AreEqual( 256.0f, result.getValue({ 0, 1 }));
			Assert::AreEqual(6561.0f, result.getValue({ 1, 0 }));
			Assert::AreEqual(   1.0f, result.getValue({ 1, 1 }));
		}

		TEST_METHOD(FitShouldDecreaseCost) {
			auto x_train = Tensor({ 128, 2 });
			auto y_train = Tensor({ 128, 2 });
			auto x_test = Tensor({ 32, 2 });
			auto y_test = Tensor({ 32, 2 });

			srand(time(NULL));

			for (uint32_t i = 0; i < x_train.getShape()[0]; ++i) {
				float x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
				float y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

				x_train.setValue(x, { i, 0 });
				x_train.setValue(y, { i, 1 });

				float u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

				y_train.setValue(u, { i, 0 });
				y_train.setValue(1.0f - u, { i, 1 });
			}
			for (uint32_t i = 0; i < x_test.getShape()[0]; ++i) {
				float x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
				float y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

				x_test.setValue(x, { i, 0 });
				x_test.setValue(y, { i, 1 });

				float u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

				y_test.setValue(u, { i, 0 });
				y_test.setValue(1.0f - u, { i, 1 });
			}

			auto layer_1 = ActivationLayer({ 2 }, ActivationFun::ReLU);
			auto layer_2 = DenseLayer(layer_1, 128);
			auto layer_3 = ActivationLayer(layer_2, ActivationFun::ReLU);
			auto layer_4 = DenseLayer(layer_3, 128);
			auto layer_5 = ActivationLayer(layer_4, ActivationFun::ReLU);
			auto layer_6 = DenseLayer(layer_5, 2);
			auto layer_7 = ActivationLayer(layer_6, ActivationFun::Sigmoid);

			auto nn = NeuralNetwork(layer_1, layer_7, CostFun::BinaryCrossentropy);

			Tensor y_hat = nn.predict(x_test);

			float cost_1 = nn.getCostFun()(y_hat, y_test);

			nn.fit(x_train, y_train, x_test, y_test, 32, 10, 0.01f);

			y_hat = nn.predict(x_test);

			float cost_2 = nn.getCostFun()(y_hat, y_test);

			Assert::IsTrue(cost_2 < cost_1);
		}
	};
}
