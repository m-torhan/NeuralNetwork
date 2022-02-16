#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetwork/Tensor.cpp"
#include "../NeuralNetwork/Layer.cpp"
#include "../NeuralNetwork/ActivationLayer.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkUnitTests {
	TEST_CLASS(TensorUnitTests) {
	public:
		
		TEST_METHOD(WhenGetValueShouldReturnProperItem) {
			Tensor* tensor;
			
			tensor = new Tensor(2, 3, 3);

			tensor->setValue(.5f, 2, 1, 2);

			Assert::AreEqual(.5f, tensor->getValue(2, 1, 2));
		}

		TEST_METHOD(WhenGetValueZeroDimensionalTensorShouldReturnNumber) {
			Tensor* tensor;

			tensor = new Tensor(0);

			tensor->setValue(1.0f, 0);

			Assert::AreEqual(1.0f, tensor->getValue(0));
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
		}

		TEST_METHOD(WhenTensorsHaveSameSizeDotProductShouldReturnSumOfAllProducts) {
			Tensor* tensor_a;
			Tensor* tensor_b;
			Tensor* dot_prod;

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

			dot_prod = tensor_a->dotProduct(*tensor_b);

			Assert::AreEqual(21.25f, dot_prod->getValue(0));
		}

		TEST_METHOD(WhenTensorsHaveDifferentSizeDotProductShouldReturnSumsOfProducts) {
			Tensor* tensor_a;
			Tensor* tensor_b;
			Tensor* dot_prod;

			tensor_a = new Tensor(2, 2, 2);
			tensor_b = new Tensor(1, 2);

			tensor_a->setValue(1.0f, 2, 0, 0);
			tensor_a->setValue(.5f, 2, 0, 1);
			tensor_a->setValue(.25f, 2, 1, 0);
			tensor_a->setValue(.125f, 2, 1, 1);

			tensor_b->setValue(16.0f, 1, 0);
			tensor_b->setValue(8.0f, 1, 1);

			dot_prod = tensor_a->dotProduct(*tensor_b);

			Assert::AreEqual(20.0f, dot_prod->getValue(1, 0));
			Assert::AreEqual(5.0f, dot_prod->getValue(1, 1));
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
		}

		TEST_METHOD(ApplyFunctionShouldApplyGivenFunctionToTensor) {
			Tensor* tensor;

			tensor = new Tensor(2, 2, 2);

			tensor->setValue(1.0f, 2, 0, 0);
			tensor->setValue(2.0f, 2, 0, 1);
			tensor->setValue(3.0f, 2, 1, 0);
			tensor->setValue(4.0f, 2, 1, 1);

			tensor->applyFunction([](float value) {return value * 2.0f; });

			Assert::AreEqual(2.0f, tensor->getValue(2, 0, 0));
			Assert::AreEqual(4.0f, tensor->getValue(2, 0, 1));
			Assert::AreEqual(6.0f, tensor->getValue(2, 1, 0));
			Assert::AreEqual(8.0f, tensor->getValue(2, 1, 1));
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
			result = layer->backwardPropagation(*tensor);

			Assert::AreEqual(  2.0f, result->getValue(2, 0, 0));
			Assert::AreEqual( 16.0f, result->getValue(2, 0, 1));
			Assert::AreEqual( 54.0f, result->getValue(2, 1, 0));
			Assert::AreEqual(128.0f, result->getValue(2, 1, 1));
		}
	};
}
