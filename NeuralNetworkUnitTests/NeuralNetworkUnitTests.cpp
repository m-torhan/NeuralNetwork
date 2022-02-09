#include "pch.h"
#include "CppUnitTest.h"
#include "../NeuralNetwork/Tensor.cpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace NeuralNetworkUnitTests
{
	TEST_CLASS(TensorUnitTests)
	{
	public:
		
		TEST_METHOD(WhenGetItemShouldReturnProperItem)
		{
			Tensor tensor = Tensor(2, 3, 3);

			tensor.setValue(.5f, 2, 1, 2);

			float ret_value = tensor.getValue(2, 1, 2);

			Assert::AreEqual(.5f, ret_value);
		}

		TEST_METHOD(WhenMultipliedByTensorEachValuePairShouldBeMultiplied)
		{
			Tensor tensor_a = Tensor(2, 2, 2);
			Tensor tensor_b = Tensor(2, 2, 2);

			tensor_a.setValue(1.0f, 2, 0, 0);
			tensor_a.setValue(.5f, 2, 0, 1);
			tensor_a.setValue(.25f, 2, 1, 0);
			tensor_a.setValue(.125f, 2, 1, 1);

			tensor_b.setValue(16.0f, 2, 0, 0);
			tensor_b.setValue(8.0f, 2, 0, 1);
			tensor_b.setValue(4.0f, 2, 1, 0);
			tensor_b.setValue(2.0f, 2, 1, 1);

			tensor_a *= &tensor_b;

			Assert::AreEqual(16.0f, tensor_a.getValue(2, 0, 0));
			Assert::AreEqual(4.0f, tensor_a.getValue(2, 0, 1));
			Assert::AreEqual(1.0f, tensor_a.getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor_a.getValue(2, 1, 1));
		}

		TEST_METHOD(WhenMultipliedByRowTensorEachValuePairShouldBeMultiplied)
		{
			Tensor tensor_a = Tensor(2, 2, 2);
			Tensor tensor_b = Tensor(1, 2);

			tensor_a.setValue(1.0f, 2, 0, 0);
			tensor_a.setValue(.5f, 2, 0, 1);
			tensor_a.setValue(.25f, 2, 1, 0);
			tensor_a.setValue(.125f, 2, 1, 1);

			tensor_b.setValue(4.0f, 1, 0);
			tensor_b.setValue(2.0f, 1, 1);

			tensor_a *= &tensor_b;

			Assert::AreEqual(4.0f, tensor_a.getValue(2, 0, 0));
			Assert::AreEqual(1.0f, tensor_a.getValue(2, 0, 1));
			Assert::AreEqual(1.0f, tensor_a.getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor_a.getValue(2, 1, 1));
		}

		TEST_METHOD(WhenMultipliedByNumberEachValueShouldBeMultiplied)
		{
			Tensor tensor = Tensor(2, 2, 2);

			tensor.setValue(1.0f, 2, 0, 0);
			tensor.setValue(.5f, 2, 0, 1);
			tensor.setValue(.25f, 2, 1, 0);
			tensor.setValue(.125f, 2, 1, 1);

			tensor *= 2;

			Assert::AreEqual(2.0f, tensor.getValue(2, 0, 0));
			Assert::AreEqual(1.0f, tensor.getValue(2, 0, 1));
			Assert::AreEqual(.5f, tensor.getValue(2, 1, 0));
			Assert::AreEqual(.25f, tensor.getValue(2, 1, 1));
		}
	};
}
