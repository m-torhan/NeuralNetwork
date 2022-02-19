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
			Tensor* tensor;
			
			tensor = new Tensor(2, 3, 3);

			tensor->setValue(.5f, 2, 1, 2);

			Assert::AreEqual(.5f, tensor->getValue(2, 1, 2));

			delete tensor;
		}

		TEST_METHOD(WhenGetValueZeroDimensionalTensorShouldReturnNumber) {
			Tensor* tensor;

			tensor = new Tensor(0);

			tensor->setValue(1.0f, 0);

			Assert::AreEqual(1.0f, tensor->getValue(0));

			delete tensor;
		}

		TEST_METHOD(WhenAddZeroDimensionalTensorsShouldBehaveLikeNumbers) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(0);
			tensor_b = new Tensor(0);

			tensor_a->setValue(6.0f, 0);
			tensor_b->setValue(2.0f, 0);

			*tensor_a += *tensor_b;

			Assert::AreEqual(8.0f, tensor_a->getValue(0));

			delete tensor_a;
			delete tensor_b;
		}

		TEST_METHOD(WhenMultiplyZeroDimensionalTensorsShouldBehaveLikeNumbers) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(0);
			tensor_b = new Tensor(0);

			tensor_a->setValue(0.5f, 0);
			tensor_b->setValue(4.0f, 0);

			*tensor_a *= *tensor_b;

			Assert::AreEqual(2.0f, tensor_a->getValue(0));

			delete tensor_a;
			delete tensor_b;
		}
		
		TEST_METHOD(SetValueShouldBeProperlyPlacedInData) {
			Tensor* tensor;

			tensor = new Tensor(3, 2, 2, 2);

			tensor->setValue(1.0f, 3, 0, 0, 0);
			tensor->setValue(2.0f, 3, 0, 0, 1);
			tensor->setValue(3.0f, 3, 0, 1, 0);
			tensor->setValue(4.0f, 3, 0, 1, 1);
			tensor->setValue(5.0f, 3, 1, 0, 0);
			tensor->setValue(6.0f, 3, 1, 0, 1);
			tensor->setValue(7.0f, 3, 1, 1, 0);
			tensor->setValue(8.0f, 3, 1, 1, 1);

			Assert::AreEqual(1.0f, tensor->getData()[0]);
			Assert::AreEqual(2.0f, tensor->getData()[1]);
			Assert::AreEqual(3.0f, tensor->getData()[2]);
			Assert::AreEqual(4.0f, tensor->getData()[3]);
			Assert::AreEqual(5.0f, tensor->getData()[4]);
			Assert::AreEqual(6.0f, tensor->getData()[5]);
			Assert::AreEqual(7.0f, tensor->getData()[6]);
			Assert::AreEqual(8.0f, tensor->getData()[7]);

			delete tensor;
		}

		TEST_METHOD(WhenMultipliedByTensorEachValuePairShouldBeMultiplied) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(2, 2, 2);
			tensor_b = new Tensor(2, 2, 2);

			tensor_a->setValue(1.0f, 2, 0, 0);
			tensor_a->setValue(.5f, 2, 0, 1);
			tensor_a->setValue(.25f, 2, 1, 0);
			tensor_a->setValue(.125f, 2, 1, 1);

			tensor_b->setValue(16.0f, 2, 0, 0);
			tensor_b->setValue(8.0f, 2, 0, 1);
			tensor_b->setValue(4.0f, 2, 1, 0);
			tensor_b->setValue(2.0f, 2, 1, 1);

			*tensor_a *= *tensor_b;

			Assert::AreEqual(16.0f, tensor_a->getValue(2, 0, 0));
			Assert::AreEqual(4.0f, tensor_a->getValue(2, 0, 1));
			Assert::AreEqual(1.0f, tensor_a->getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor_a->getValue(2, 1, 1));

			delete tensor_a;
			delete tensor_b;
		}

		TEST_METHOD(WhenMultipliedByRowTensorEachValuePairShouldBeMultiplied) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(2, 2, 2);
			tensor_b = new Tensor(1, 2);

			tensor_a->setValue(1.0f, 2, 0, 0);
			tensor_a->setValue(.5f, 2, 0, 1);
			tensor_a->setValue(.25f, 2, 1, 0);
			tensor_a->setValue(.125f, 2, 1, 1);

			tensor_b->setValue(4.0f, 1, 0);
			tensor_b->setValue(2.0f, 1, 1);

			*tensor_a *= *tensor_b;

			Assert::AreEqual(4.0f, tensor_a->getValue(2, 0, 0));
			Assert::AreEqual(1.0f, tensor_a->getValue(2, 0, 1));
			Assert::AreEqual(1.0f, tensor_a->getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor_a->getValue(2, 1, 1));

			delete tensor_a;
			delete tensor_b;
		}

		TEST_METHOD(WhenMultipliedByNumberEachValueShouldBeMultiplied) {
			Tensor* tensor;

			tensor = new Tensor(2, 2, 2);

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(.5f, 2, 0, 1);
			tensor->setValue(.25f, 2, 1, 0);
			tensor->setValue(.125f, 2, 1, 1);

			*tensor *= 2;

			Assert::AreEqual(2.0f, tensor->getValue(2, 0, 0));
			Assert::AreEqual(1.0f, tensor->getValue(2, 0, 1));
			Assert::AreEqual(.5f, tensor->getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor->getValue(2, 1, 1));

			delete tensor;
		}

		TEST_METHOD(WhenTensorsAreMatrixAndVectorDotProductShouldReturnSumsOfProductsOfRowsByVector) {
			Tensor* tensor_a;
			Tensor* tensor_b;
			Tensor* result;

			tensor_a = new Tensor(2, 2, 2);
			tensor_b = new Tensor(1, 2);

			tensor_a->setValue(1.0f, 2, 0, 0);
			tensor_a->setValue(.5f, 2, 0, 1);
			tensor_a->setValue(.25f, 2, 1, 0);
			tensor_a->setValue(.125f, 2, 1, 1);

			tensor_b->setValue(16.0f, 1, 0);
			tensor_b->setValue(8.0f, 1, 1);

			result = tensor_a->dotProduct(*tensor_b);

			Assert::AreEqual(18.0f, result->getValue(1, 0));
			Assert::AreEqual( 9.0f, result->getValue(1, 1));

			delete tensor_a;
			delete tensor_b;
			delete result;
		}

		TEST_METHOD(TensorProductResultDimShouldBeSumOfArgumentsDims) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(2, 2, 3);
			tensor_b = new Tensor(3, 4, 5, 6);

			Tensor* result = tensor_a->tensorProduct(*tensor_b);

			Assert::AreEqual(5, (int)result->getDim());
		}

		TEST_METHOD(TensorProductResultShapeShouldBeConcatOfArgumentsShapes) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(2, 2, 3);
			tensor_b = new Tensor(3, 4, 5, 6);

			Tensor* result = tensor_a->tensorProduct(*tensor_b);

			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(3, (int)result->getShape()[1]);
			Assert::AreEqual(4, (int)result->getShape()[2]);
			Assert::AreEqual(5, (int)result->getShape()[3]);
			Assert::AreEqual(6, (int)result->getShape()[4]);

			delete tensor_a;
			delete tensor_b;
			delete result;
		}

		TEST_METHOD(TensorProductResultShouldBeCorrect) {
			Tensor* tensor_a;
			Tensor* tensor_b;

			tensor_a = new Tensor(2, 2, 3);
			tensor_b = new Tensor(3, 4, 5, 6);

			*tensor_a *= .0f;
			*tensor_b *= .0f;

			tensor_a->setValue(2.0f, 2, 0, 0);
			tensor_a->setValue(4.0f, 2, 0, 1);
			tensor_a->setValue(8.0f, 2, 0, 2);
			tensor_a->setValue(16.0f, 2, 1, 0);
			tensor_a->setValue(32.0f, 2, 1, 1);
			tensor_a->setValue(64.0f, 2, 1, 2);

			tensor_b->setValue(.5f, 3, 0, 0, 0);
			tensor_b->setValue(.25f, 3, 1, 2, 3);
			tensor_b->setValue(.125f, 3, 1, 2, 0);

			Tensor* result = tensor_a->tensorProduct(*tensor_b);

			Assert::AreEqual(1.0f, result->getValue(5, 0, 0, 0, 0, 0));
			Assert::AreEqual(32.0f, result->getValue(5, 1, 2, 0, 0, 0));
			Assert::AreEqual(1.0f, result->getValue(5, 0, 1, 1, 2, 3));
			Assert::AreEqual(8.0f, result->getValue(5, 1, 1, 1, 2, 3));
			Assert::AreEqual(1.0f, result->getValue(5, 0, 2, 1, 2, 0));
			Assert::AreEqual(2.0f, result->getValue(5, 1, 0, 1, 2, 0));

			delete tensor_a;
			delete tensor_b;
			delete result;
		}

		TEST_METHOD(ApplyFunctionShouldApplyGivenFunctionToTensor) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(2, 2, 2);

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);

			result = tensor->applyFunction([](float value) {return value * 2.0f; });

			Assert::AreEqual(2.0f, result->getValue(2, 0, 0));
			Assert::AreEqual(4.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(6.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(8.0f, result->getValue(2, 1, 1));

			delete tensor;
			delete result;
		}

		TEST_METHOD(SumAcrossFirstAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->sum(0);

			Assert::AreEqual(2, (int)result->getDim());

			Assert::AreEqual( 8.0f, result->getValue(2, 0, 0));
			Assert::AreEqual(10.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(12.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(14.0f, result->getValue(2, 1, 1));
			Assert::AreEqual(16.0f, result->getValue(2, 2, 0));
			Assert::AreEqual(18.0f, result->getValue(2, 2, 1));

			delete tensor;
			delete result;
		}

		TEST_METHOD(SumAcrossSecondAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->sum(1);

			Assert::AreEqual(2, (int)result->getDim());

			Assert::AreEqual( 9.0f, result->getValue(2, 0, 0));
			Assert::AreEqual(12.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(27.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(30.0f, result->getValue(2, 1, 1));

			delete tensor;
			delete result;
		}

		TEST_METHOD(SumAcrossThridAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->sum(2);

			Assert::AreEqual(2, (int)result->getDim());

			Assert::AreEqual( 3.0f, result->getValue(2, 0, 0));
			Assert::AreEqual( 7.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(11.0f, result->getValue(2, 0, 2));
			Assert::AreEqual(15.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(19.0f, result->getValue(2, 1, 1));
			Assert::AreEqual(23.0f, result->getValue(2, 1, 2));

			delete tensor;
			delete result;
		}

		TEST_METHOD(WhenFlattenResultShouldHaveOnlyOneDimEqualToSize) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			result = tensor->flatten();

			Assert::AreEqual(1, (int)result->getDim());
			Assert::AreEqual(tensor->getSize(), result->getShape()[0]);

			delete tensor;
			delete result;
		}

		TEST_METHOD(WhenFlattenFromFirstAxisResultShouldBeTwoDimensionalAndFirstShapeShouldRemain) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			result = tensor->flatten(1);

			Assert::AreEqual(2, (int)result->getDim());
			Assert::AreEqual(tensor->getShape()[0], result->getShape()[0]);
			Assert::AreEqual(tensor->getSize()/tensor->getShape()[0], result->getShape()[1]);

			delete tensor;
			delete result;
		}

		TEST_METHOD(TensorSliceFirstAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->slice(0, 1, 2);

			Assert::AreEqual(3, (int)result->getDim());

			Assert::AreEqual(1, (int)result->getShape()[0]);
			Assert::AreEqual(3, (int)result->getShape()[1]);
			Assert::AreEqual(2, (int)result->getShape()[2]);

			Assert::AreEqual( 7.0f, result->getValue(3, 0, 0, 0));
			Assert::AreEqual( 8.0f, result->getValue(3, 0, 0, 1));
			Assert::AreEqual( 9.0f, result->getValue(3, 0, 1, 0));
			Assert::AreEqual(10.0f, result->getValue(3, 0, 1, 1));
			Assert::AreEqual(11.0f, result->getValue(3, 0, 2, 0));
			Assert::AreEqual(12.0f, result->getValue(3, 0, 2, 1));

			delete tensor;
			delete result;
		}

		TEST_METHOD(TensorSliceSecondAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->slice(1, 1, 2);

			Assert::AreEqual(3, (int)result->getDim());

			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(1, (int)result->getShape()[1]);
			Assert::AreEqual(2, (int)result->getShape()[2]);

			Assert::AreEqual( 3.0f, result->getValue(3, 0, 0, 0));
			Assert::AreEqual( 4.0f, result->getValue(3, 0, 0, 1));
			Assert::AreEqual( 9.0f, result->getValue(3, 1, 0, 0));
			Assert::AreEqual(10.0f, result->getValue(3, 1, 0, 1));

			delete tensor;
			delete result;
		}

		TEST_METHOD(TensorSliceThirdAxis) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(3, 2, 3, 2);

			tensor->setValue( 1.0f, 3, 0, 0, 0);
			tensor->setValue( 2.0f, 3, 0, 0, 1);
			tensor->setValue( 3.0f, 3, 0, 1, 0);
			tensor->setValue( 4.0f, 3, 0, 1, 1);
			tensor->setValue( 5.0f, 3, 0, 2, 0);
			tensor->setValue( 6.0f, 3, 0, 2, 1);
			tensor->setValue( 7.0f, 3, 1, 0, 0);
			tensor->setValue( 8.0f, 3, 1, 0, 1);
			tensor->setValue( 9.0f, 3, 1, 1, 0);
			tensor->setValue(10.0f, 3, 1, 1, 1);
			tensor->setValue(11.0f, 3, 1, 2, 0);
			tensor->setValue(12.0f, 3, 1, 2, 1);

			result = tensor->slice(2, 1, 2);

			Assert::AreEqual(3, (int)result->getDim());

			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(3, (int)result->getShape()[1]);
			Assert::AreEqual(1, (int)result->getShape()[2]);

			Assert::AreEqual( 2.0f, result->getValue(3, 0, 0, 0));
			Assert::AreEqual( 4.0f, result->getValue(3, 0, 1, 0));
			Assert::AreEqual( 6.0f, result->getValue(3, 0, 2, 0));
			Assert::AreEqual( 8.0f, result->getValue(3, 1, 0, 0));
			Assert::AreEqual(10.0f, result->getValue(3, 1, 1, 0));
			Assert::AreEqual(12.0f, result->getValue(3, 1, 2, 0));

			delete tensor;
			delete result;
		}

		TEST_METHOD(TensorShuffleShouldRearrangeValues) {
			Tensor* tensor;
			Tensor* result;

			tensor = new Tensor(2, 2, 2);

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);

			result = tensor->shuffle();

			Assert::IsTrue(result->getValue(2, 0, 0) == 1 || result->getValue(2, 1, 0) == 1);
			Assert::IsTrue(result->getValue(2, 0, 0) == 3 || result->getValue(2, 1, 0) == 3);
			Assert::IsTrue(result->getValue(2, 0, 1) == 2 || result->getValue(2, 1, 1) == 2);
			Assert::IsTrue(result->getValue(2, 0, 1) == 4 || result->getValue(2, 1, 1) == 4);

			delete tensor;
			delete result;
		}

		TEST_METHOD(TensorShuffleWithPatternShouldRearrangeValues) {
			Tensor* tensor;
			Tensor* result;
			uint32_t pattern[4] = { 2, 3, 0, 1 };

			tensor = new Tensor(2, 4, 2);

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);
			tensor->setValue(5.0f, 2, 2, 0);
			tensor->setValue(6.0f, 2, 2, 1);
			tensor->setValue(7.0f, 2, 3, 0);
			tensor->setValue(8.0f, 2, 3, 1);

			result = tensor->shuffle(pattern);
			
			Assert::AreEqual(5.0f, result->getValue(2, 0, 0));
			Assert::AreEqual(7.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(1.0f, result->getValue(2, 2, 0));
			Assert::AreEqual(3.0f, result->getValue(2, 3, 0));
		}
	};

	TEST_CLASS(ActivationLayerUnitTests) {
	public:

		TEST_METHOD(ActivationLayerShouldApplyGivenFunctionWhenPrograpateForward) {
			Tensor* tensor;
			Tensor* result;
			ActivationLayer* layer;

			tensor = new Tensor(2, 2, 2);
			layer = new ActivationLayer(tensor->getDim(), tensor->getShape(), [](const Tensor& x) { return x * x; }, [](const Tensor& x, const Tensor& dx) { return x * (*(dx * 2.0f)); });

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);

			result = layer->forwardPropagation(*tensor);

			Assert::AreEqual( 1.0f, result->getValue(2, 0, 0));
			Assert::AreEqual( 4.0f, result->getValue(2, 0, 1));
			Assert::AreEqual( 9.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(16.0f, result->getValue(2, 1, 1));

			delete tensor;
			delete result;
			delete layer;
		}

		TEST_METHOD(ActivationLayerShouldApplyGivenFunctionDerivativeWhenPrograpateBackward) {
			Tensor* tensor;
			Tensor* result;
			ActivationLayer* layer;

			tensor = new Tensor(2, 2, 2);
			layer = new ActivationLayer(tensor->getDim(), tensor->getShape(), [](const Tensor& x) { return x * x; }, [](const Tensor& x, const Tensor& dx) { return x * (*(dx * 2.0f)); });

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);

			result = layer->forwardPropagation(*tensor);
			result = layer->backwardPropagation(*tensor, 1.0f);

			Assert::AreEqual( 2.0f, result->getValue(2, 0, 0));
			Assert::AreEqual( 8.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(18.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(32.0f, result->getValue(2, 1, 1));

			delete tensor;
			delete result;
			delete layer;
		}
	};
	TEST_CLASS(DenseLayerUnitTests) {
	public:
		TEST_METHOD(DenseLayerForwardPropagationOutputShapeTest) {
			Tensor* tensor;
			Tensor* result;
			DenseLayer* layer;
			uint32_t input_shape[2] = { 3, 4 };

			tensor = new Tensor(3, 2, 3, 4);
			layer = new DenseLayer(2, input_shape, 4);

			result = layer->forwardPropagation(*tensor);

			Assert::AreEqual(2, (int)result->getDim());
			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(4, (int)result->getShape()[1]);

			delete tensor;
			delete result;
			delete layer;
		}

		TEST_METHOD(DenseLayerBackwardPropagationOutputShapeTest) {
			Tensor* tensor;
			Tensor* tensor_d;
			Tensor* result;
			DenseLayer* layer;
			uint32_t input_shape[2] = { 3, 4 };

			tensor = new Tensor(3, 2, 3, 4);
			tensor_d = new Tensor(2, 2, 4);
			layer = new DenseLayer(2, input_shape, 4);

			layer->initCachedGradient();
			layer->forwardPropagation(*tensor);
			result = layer->backwardPropagation(*tensor_d, 1.0f);

			Assert::AreEqual(2, (int)result->getDim());
			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(12, (int)result->getShape()[1]);

			delete tensor;
			delete tensor_d;
			delete result;
			delete layer;
		}
	};

	TEST_CLASS(NeuralNetworkUnitTests) {
		TEST_METHOD(PredictShouldReturnTensor) {
			Tensor* tensor;
			Tensor* result;
			Tensor* (*activation_fun)(const Tensor & x) = [](const Tensor& x) { return x * x; };
			Tensor* (*activation_fun_d)(const Tensor & x, const Tensor & dx) = [](const Tensor& x, const Tensor& dx) { return x * (*(dx * 2.0f)); };
			ActivationLayer* layer_1;
			ActivationLayer* layer_2;
			ActivationLayer* layer_3;
			NeuralNetwork* nn;

			tensor = new Tensor(2, 2, 2);
			tensor->setValue( 1.0f, 2, 0, 0);
			tensor->setValue( 2.0f, 2, 0, 1);
			tensor->setValue( 3.0f, 2, 1, 0);
			tensor->setValue(-1.0f, 2, 1, 1);

			layer_1 = new ActivationLayer(tensor->getDim(), tensor->getShape(), activation_fun, activation_fun_d);
			layer_2 = new ActivationLayer(*layer_1, activation_fun, activation_fun_d);
			layer_3 = new ActivationLayer(*layer_2, activation_fun, activation_fun_d);

			nn = new NeuralNetwork(*layer_1, *layer_3, nullptr, nullptr);

			result = nn->predict(tensor);

			Assert::AreEqual(2, (int)result->getDim());
			Assert::AreEqual(2, (int)result->getShape()[0]);
			Assert::AreEqual(2, (int)result->getShape()[1]);
			Assert::AreEqual(   1.0f, result->getValue(2, 0, 0));
			Assert::AreEqual( 256.0f, result->getValue(2, 0, 1));
			Assert::AreEqual(6561.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(   1.0f, result->getValue(2, 1, 1));

			delete tensor;
			delete result;
			delete layer_1;
			delete layer_2;
			delete layer_3;
			delete nn;
		}

		TEST_METHOD(FitShouldDecreaseCost) {
			uint32_t i = 0;
			Tensor* x_train;
			Tensor* y_train;
			Tensor* x_test;
			Tensor* y_test;
			Tensor* y_hat;
			Tensor* (*activation_fun)(const Tensor & x) = [](const Tensor& x) { return x * x; };
			Tensor* (*activation_fun_d)(const Tensor & x, const Tensor & dx) = [](const Tensor& x, const Tensor& dx) { return x * (*(dx * 2.0f)); };
			ActivationLayer* layer_1;
			DenseLayer* layer_2;
			ActivationLayer* layer_3;
			DenseLayer* layer_4;
			ActivationLayer* layer_5;
			DenseLayer* layer_6;
			ActivationLayer* layer_7;
			NeuralNetwork* nn;
			float x;
			float y;
			float u;
			float cost_1;
			float cost_2;

			x_train = new Tensor(2, 128, 2);
			y_train = new Tensor(2, 128, 2);
			x_test = new Tensor(2, 32, 2);
			y_test = new Tensor(2, 32, 2);

			srand(time(NULL));

			for (i = 0; i < x_train->getShape()[0]; ++i) {
				x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
				y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

				x_train->setValue(x, 2, i, 0);
				x_train->setValue(y, 2, i, 1);

				u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

				y_train->setValue(u, 2, i, 0);
				y_train->setValue(1.0f - u, 2, i, 1);
			}
			for (i = 0; i < x_test->getShape()[0]; ++i) {
				x = static_cast<float>(rand() % 256) / 128.0f - 1.0f;
				y = static_cast<float>(rand() % 256) / 128.0f - 1.0f;

				x_test->setValue(x, 2, i, 0);
				x_test->setValue(y, 2, i, 1);

				u = (x * x + y * y < 0.798f * 0.798f) ? 1.0f : 0.0f;

				y_test->setValue(u, 2, i, 0);
				y_test->setValue(1.0f - u, 2, i, 1);
			}

			layer_1 = new ActivationLayer(x_train->getDim() - 1, &x_train->getShape()[1], ActivationFun::ReLU);
			layer_2 = new DenseLayer(*layer_1, 128);
			layer_3 = new ActivationLayer(*layer_2, ActivationFun::ReLU);
			layer_4 = new DenseLayer(*layer_3, 128);
			layer_5 = new ActivationLayer(*layer_4, ActivationFun::ReLU);
			layer_6 = new DenseLayer(*layer_5, 2);
			layer_7 = new ActivationLayer(*layer_6, ActivationFun::Sigmoid);

			nn = new NeuralNetwork(*layer_1, *layer_7, CostFun::BinaryCrossentropy);

			y_hat = nn->predict(x_test);

			cost_1 = nn->getCostFun()(*y_hat, *y_test);

			nn->fit(x_train, y_train, x_test, y_test, 32, 20, 0.01f);

			y_hat = nn->predict(x_test);

			cost_2 = nn->getCostFun()(*y_hat, *y_test);

			Assert::IsTrue(cost_2 < cost_1);

			delete x_train;
			delete y_train;
			delete y_hat;
			delete layer_1;
			delete layer_2;
			delete layer_3;
			delete layer_4;
			delete layer_5;
			delete nn;
		}
	};
}
