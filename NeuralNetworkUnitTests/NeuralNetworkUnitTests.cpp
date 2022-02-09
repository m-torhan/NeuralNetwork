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
	};
}
