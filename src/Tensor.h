#pragma once

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <algorithm>
#include <cstdio>
#include <thread>

#include "Utils.h"

#ifdef SSE
	#ifndef WIN
		#define SSE_vector_inner_product 			_SSE_vector_inner_product

		#define SSE_vector_add 						_SSE_vector_add
		#define SSE_tensor_add 						_SSE_tensor_add
		#define SSE_tensor_add_scalar 				_SSE_tensor_add_scalar

		#define SSE_vector_sub 						_SSE_vector_sub
		#define SSE_tensor_sub 						_SSE_tensor_sub
		#define SSE_tensor_sub_scalar 				_SSE_tensor_sub_scalar
		#define SSE_scalar_sub_tensor 				_SSE_scalar_sub_tensor

		#define SSE_vector_mul 						_SSE_vector_mul
		#define SSE_tensor_mul 						_SSE_tensor_mul
		#define SSE_tensor_mul_scalar 				_SSE_tensor_mul_scalar

		#define SSE_vector_div 						_SSE_vector_div
		#define SSE_tensor_div 						_SSE_tensor_div
		#define SSE_tensor_div_scalar 				_SSE_tensor_div_scalar
		#define SSE_scalar_div_tensor 				_SSE_scalar_div_tensor
		
		#define SSE_tensor_sum 						_SSE_tensor_sum
		#define SSE_tensor_axis_sum					_SSE_tensor_axis_sum
		#define SSE_tensor_last_axis_sum			_SSE_tensor_last_axis_sum

		#define SSE_tensor_dot_product_transpose	_SSE_tensor_dot_product_transpose
	#endif
	
	extern "C" {
		void SSE_vector_inner_product(const uint32_t n, const float* v1, const float* v2, float* r);

		void SSE_vector_add(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_add(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_add_scalar(const uint32_t n, const float* v, const float* s, float* r);
		
		void SSE_vector_sub(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_sub(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_sub_scalar(const uint32_t n, const float* v, const float* s, float* r);
		void SSE_scalar_sub_tensor(const float* s, const uint32_t n, const float* v, float* r);

		void SSE_vector_mul(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_mul(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_mul_scalar(const uint32_t n, const float* v, const float* s, float* r);

		void SSE_vector_div(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_div(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_div_scalar(const uint32_t n, const float* v, const float* s, float* r);
		void SSE_scalar_div_tensor(const float* s, const uint32_t n, const float* v, float* r);
		
		void SSE_tensor_sum(const uint32_t n, const float* v, float* r);
		void SSE_tensor_axis_sum(const uint32_t n, const uint32_t m, const uint32_t k, const float* v, float* r);
		void SSE_tensor_last_axis_sum(const uint32_t n, const uint32_t k, const float* v, float* r);

		void SSE_tensor_dot_product_transpose(const uint32_t n, const uint32_t m, const uint32_t k, const float* v1, const float *v2, float *r);
	}
#endif

enum Padding : uint8_t {
	Left = 0x01,
	Right = 0x02,
	Both = 0x03,
};

class Tensor;

/**
 * @brief TensorSlice class used for tensor slice assignment.
 */
class TensorSlice {
public:
    /**
     * Assigns a tensor values to the slice.
     * @brief assign operator.
     * @param other The tensor which values are used.
     */
	const Tensor& operator=(const Tensor& other);

private:
	TensorSlice() = delete;
    /**
     * Creates a new TensorSlice object for given tensor and ranges.
     * @brief constructor.
     * @param tensor The tensor which slice is represented by TensorSlice.
     * @param slice_ranges Ranges of selected slice.
     */
	TensorSlice(Tensor& tensor, std::vector<std::vector<uint32_t> > slice_ranges) :
		_tensor(tensor), _slice_ranges(slice_ranges) {}

    /**
     * The tensor which slice is represented by TensorSlice.
     */ 
	Tensor& _tensor;
    /**
     * Ranges of selected slice.
     */ 
	std::vector<std::vector<uint32_t> > _slice_ranges;
	
	friend class Tensor;
};

/**
 * @brief TensorCell class used for tensor cell assignment.
 */
class TensorCell {
public:
    /**
     * Assigns a float value to the cell.
     * @brief assign operator.
     * @param other The value which is used.
     */
	float operator=(float value);

private:
	TensorCell() = delete;
    /**
     * Creates a new TensorCell object for given tensor and index.
     * @brief constructor.
     * @param tensor The tensor which cell is represented by TensorCell.
     * @param cell_index Index of selected tensor cell.
     */
	TensorCell(Tensor& tensor, std::vector<uint32_t> cell_index) :
		_tensor(tensor), _cell_index(cell_index) {}

    /**
     * The tensor which cell is represented by TensorCell.
     */ 
	Tensor& _tensor;
    /**
     * Index of selected tensor cell.
     */ 
	std::vector<uint32_t> _cell_index;

	friend class Tensor;
};

/**
 * @brief Tensor class that represents mathematical tensor.
 */
class Tensor {
public:
    /**
     * Types used for selecting tesnor cell and slice.
     */ 
	typedef std::vector<uint32_t> Index;
	typedef std::vector<std::vector<uint32_t>> Range;
	
    /**
     * Create a new Tensor of shape (1) and value 0.0f, (a scalar).
     * @brief Default constructor.
     */
	Tensor();
    /**
     * Creates a new Tensor object of given shape and values 0.0f.
     * @brief constructor.
	 * 
     * @param shape Shape of the tensor.
     */
	Tensor(const std::vector<uint32_t>& shape);
    /**
     * Construct a new Tensor object from another Tensor object.
     * @brief Copy constructor.
	 * 
     * @param other Another Tensor object.
     */
	Tensor(const Tensor& other);
    /**
     * Copies values from another Tensor objects.
     * @brief Assign operator.
	 * 
     * @param other Another Tensor object.
     */
	const Tensor& operator=(const Tensor& other);
    /**
     * Moves values from another Tensor objects.
     * @brief Move operator.
	 * 
     * @param other Another Tensor object.
     */
	const Tensor& operator=(const Tensor&& other);

    /**
     * Creates a new Tensor object of given shape and values from normal distribution.
	 * @brief Create Tensor with values from Normal Distribution.
	 * 
     * @param shape Shape of the tensor.
	 * @return Tensor with values sampled from Normal Distribution.
     */
	static Tensor RandomNormal(const std::vector<uint32_t>& shape);

    /**
     * Returns a slice of Tensor as a Tensor (rvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Ranges of selected slice.
	 * @return Slice as a Tensor.
     */
	const Tensor operator[](std::vector<std::vector<uint32_t> > ranges) const;
    /**
     * Returns a slice of Tensor as a TensorSlice (lvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Ranges of selected slice.
	 * @return Slice as a TensorSlice.
     */
	TensorSlice operator[](std::vector<std::vector<uint32_t> > ranges);
    /**
     * Returns a cell of Tensor as a float (rvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Index of selected cell.
	 * @return Cell value as float.
     */
	const float operator[](const std::vector<uint32_t>& index) const;
    /**
     * Returns a cell of Tensor as a float (rvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Index of selected cell.
	 * @return Cell value as a float.
     */
	const float operator[](const std::initializer_list<uint32_t>& ini) const
	{
		return operator[](std::vector<uint32_t>(ini));
	}
    /**
     * Returns a cell of Tensor as a TensorCell (lvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Index of selected cell.
	 * @return Cell value as a TensorCell.
     */
	TensorCell operator[](const std::vector<uint32_t>& index);
    /**
     * Returns a cell of Tensor as a TensorCell (lvalue).
	 * @brief Subscript operator.
	 * 
     * @param ranges Index of selected cell.
	 * @return Cell value as a TensorCell.
     */
	TensorCell operator[](const std::initializer_list<uint32_t>& ini)
	{
		return operator[](std::vector<uint32_t>(ini));
	}

    /**
     * @brief Get shape of the Tensor.
	 * 
	 * @return Shape of the tensor.
     */
	std::vector<uint32_t> getShape() const;
    /**
     * @brief Get dimenstion of the Tensor.
	 * 
	 * @return Dimension of the tensor.
     */
	uint32_t getDim() const;
    /**
     * @brief Get size of the Tensor.
	 * 
	 * @return Size of the tensor.
     */
	uint32_t getSize() const;
    /**
     * @brief Get all values of the Tensor.
	 * 
	 * @return Values of the tensor.
     */
	std::vector<float> getData() const;

    /**
     * @brief Sets all values of the Tensor.
	 * 
     * @param values vector of values to be set.
     */
	void setValues(const std::vector<float>& values);

    /**
     * @brief Minus operator overloading.
	 * 
     * @return Tensor with values with opposite sign.
     */
	const Tensor operator-() const;

    /**
	 * Adds tensors element-wise and store values is left operand.
     * @brief Addition assignment operator overloading.
	 * 
	 * @param other Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator+=(const Tensor& other);
    /**
     * Adds tensors element-wise and return result as a new tensor.
     * @brief Addition operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise sum of operands.
     */
	const Tensor operator+(const Tensor& other) const;
    /**
     * Adds tensor and float element-wise and store values is left operand.
     * @brief Addition assignment operator overloading.
	 * 
	 * @param number Right operand.
     * @return reference to left operand.
     */
	Tensor& operator+=(float number);
    /**
     * Adds tensor and float element-wise and return result as a new tensor.
     * @brief Addition operator overloading.
	 * 
	 * @param number Right operand.
     * @return Element-wise sum of operands.
     */
	const Tensor operator+(float number) const;
    /**
     * Adds float and tensor element-wise and return result as a new tensor.
     * @brief Addition operator overloading.
	 * 
	 * @param number Left operand.
	 * @param other Right operand.
     * @return Element-wise sum of operands.
     */
	friend const Tensor operator+(float number, const Tensor& other);

    /**
     * Subtracts tensors element-wise and store values is left operand.
     * @brief Subtraction assignment operator overloading.
	 * 
	 * @param other Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator-=(const Tensor& other);
    /**
     * Subtracts tensors element-wise and return result as a new tensor.
     * @brief Subtraction operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise difference of operands.
     */
	const Tensor operator-(const Tensor& other) const;
    /**
     * Subtracts float from tensor element-wise and store values is left operand.
     * @brief Subtraction assignment operator overloading.
	 * 
	 * @param number Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator-=(float number);
    /**
     * Subtracts float from tensor element-wise and return result as a new tensor.
     * @brief Subtraction operator overloading.
	 * 
	 * @param number Right operand.
     * @return Element-wise difference of operands.
     */
	const Tensor operator-(float number) const;
    /**
     * Subtracts tensor from float element-wise and return result as a new tensor.
     * @brief Subtraction operator overloading.
	 * 
	 * @param number Left operand.
	 * @param other Right operand.
     * @return Element-wise difference of operands.
     */
	friend const Tensor operator-(float number, const Tensor& other);

    /**
     * Multiplies tensors element-wise and store values is left operand.
     * @brief Multiplication assignment operator overloading.
	 * 
	 * @param other Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator*=(const Tensor& other);
    /**
     * Multiplies tensors element-wise and return result as a new tensor.
     * @brief Multiplication operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise product of operands.
     */
	const Tensor operator*(const Tensor& other) const;
    /**
     * Multiplies tensor and float element-wise and store values is left operand.
     * @brief Multiplication assignment operator overloading.
	 * 
	 * @param number Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator*=(float number);
    /**
     * Multiplies tensor and float element-wise and return result as a new tensor.
     * @brief Multiplication operator overloading.
	 * 
	 * @param number Right operand.
     * @return Element-wise product of operands.
     */
	const Tensor operator*(float number) const;
    /**
     * Multiplies float and tensor element-wise and return result as a new tensor.
     * @brief Multiplication operator overloading.
	 * 
	 * @param number Left operand.
	 * @param other Right operand.
     * @return Element-wise product of operands.
     */
	friend const Tensor operator*(float number, const Tensor& other);

    /**
     * Divides tensors element-wise and store values is left operand.
     * @brief Division assignment operator overloading.
	 * 
	 * @param other Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator/=(const Tensor& other);
    /**
     * Divides tensors element-wise and return result as a new tensor.
     * @brief Division operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise quotient of operands.
     */
	const Tensor operator/(const Tensor& other) const;
    /**
     * Divides tensor by float element-wise and store values is left operand.
     * @brief Division assignment operator overloading.
	 * 
	 * @param number Right operand.
     * @return Reference to left operand.
     */
	Tensor& operator/=(float number);
    /**
     * Divides tensor by float element-wise and return result as a new tensor.
     * @brief Division operator overloading.
	 * 
	 * @param number Right operand.
     * @return Element-wise quotient of operands.
     */
	const Tensor operator/(float number) const;
    /**
     * Divides float by tensor element-wise and return result as a new tensor.
     * @brief Division operator overloading.
	 * 
	 * @param number Left operand.
	 * @param other Right operand.
     * @return Element-wise quotient of operands.
     */
	friend const Tensor operator/(float number, const Tensor& other);

    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when values are equal and 0.0f in other case.
	 * @brief Equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise equal operator result of operands.
     */
	const Tensor operator==(const Tensor& other) const;
    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when values are not equal and 0.0f in other case.
	 * @brief Not equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise not equal operator result of operands.
     */
	const Tensor operator!=(const Tensor& other) const;
    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when value of left operand is greater and 0.0f in other case.
	 * @brief Greater operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise greater operator result of operands.
     */
	const Tensor operator>(const Tensor& other) const;
    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when value of left operand is greater or equal and 0.0f in other case.
	 * @brief Greater or equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise greater or equal operator result of operands.
     */
	const Tensor operator>=(const Tensor& other) const;
    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when value of left operand is less and 0.0f in other case.
	 * @brief Less operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise less operator result of operands.
     */
	const Tensor operator<(const Tensor& other) const;
    /**
     * Compares tensors element-wise and return result as a new tensor where 1.0f is set when value of left operand is less or equal and 0.0f in other case.
	 * @brief Less or equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise less or equal operator result of operands.
     */
	const Tensor operator<=(const Tensor& other) const;
	
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when values are equal and 0.0f in other case.
	 * @brief Equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise equal operator result of operands.
     */
	const Tensor operator==(float other) const;
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when values are not equal and 0.0f in other case.
	 * @brief Not equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise not equal operator result of operands.
     */
	const Tensor operator!=(float other) const;
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when value of tensor is greater and 0.0f in other case.
	 * @brief Greater operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise greater operator result of operands.
     */
	const Tensor operator>(float other) const;
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when value of tensor is greater or equal and 0.0f in other case.
	 * @brief Greater or equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise greater or equal operator result of operands.
     */
	const Tensor operator>=(float other) const;
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when value of tensor is less and 0.0f in other case.
	 * @brief Less operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise less operator result of operands.
     */
	const Tensor operator<(float other) const;
    /**
     * Compares tensor values element-wise with float and return result as a new tensor where 1.0f is set when value of tensor is less or equal and 0.0f in other case.
	 * @brief Less or equal operator overloading.
	 * 
	 * @param other Right operand.
     * @return Element-wise less or equal operator result of operands.
     */
	const Tensor operator<=(float other) const;
	
	/**
	 * @brief Adds padding to the Tensor.
	 * 
	 * @param axes Axes along which the padding will be added.
	 * @param paddings Types of paddings (Left, Right or Both).
	 * @param counts Lengths of paddings.
	 * @return Tensor with added paddings.
	 */
	const Tensor addPadding(const std::vector<uint32_t>& axes, const std::vector<Padding>& paddings, const std::vector<uint32_t>& counts) const;
	/**
	 * @brief Computes dot product of two operands.
	 * 
	 * @param other Another Tensor object.
	 * @return Dot product result.
	 */
	const Tensor dotProduct(const Tensor& other) const;
	/**
	 * @brief Computes dot product of left operand and transposition of right operand.
	 * 
	 * @param other Another Tensor object.
	 * @return Dot product result.
	 */
	const Tensor dotProductTranspose(const Tensor& other) const;
	/**
	 * @brief Computes tensor product of two operands.
	 * 
	 * @param other Another Tensor object.
	 * @return Tensor product result.
	 */
	const Tensor tensorProduct(const Tensor& other) const;
	/**
	 * @brief Applies function element-wise.
	 * 
	 * @param function Function ot be applied.
	 * @return Tensor that contains results of the function.
	 */
	const Tensor applyFunction(float (*function)(float)) const;
	/**
	 * Reduces Tensor dimension so that all dimensions starting from from_axis whill be one flatted to one dimension.
	 * @brief Reshapes Tensor to (from_axis + 1)-dim Tensor.
	 * 
	 * @param from_axis Axis from which dimensions will be reduced.
	 * @return Flattening result.
	 */
	const Tensor flatten(uint32_t from_axis=0) const;
	/**
	 * @brief Sums Tensor values along given axis.
	 * 
	 * @param axis Selected axis.
	 * @return Tensor adter applying sum along axis.
	 */
	const Tensor sum(uint32_t axis) const;
	/**
	 * @brief Sums all tensor values.
	 * 
	 * @return Sum result.
	 */
	float sum() const;
	/**
	 * @brief Finds maximum value of the Tensor.
	 * 
	 * @return Maximum value of the Tensor.
	 */
	float max() const;
	/**
	 * @brief Finds minimum value of the Tensor.
	 * 
	 * @return Minimum value of the Tensor.
	 */
	float min() const;
	/**
	 * @brief Computes mean value of the Tensor.
	 * 
	 * @return Mean value of the Tensor.
	 */
	float mean() const;
	/**
	 * @brief Trasposes the Tensor.
	 * 
	 * @return Transposed Tensor.
	 */
	const Tensor transpose() const;
	/**
	 * @brief Shuffles Tensor values along first axis.
	 * 
	 * @return Shuffling result.
	 */
	const Tensor shuffle() const;
	/**
	 * @brief Shuffles Tensor values according to given pattern along first axis.
	 * 
	 * @param pattern Shuffling pattern.
	 * @return const Tensor 
	 */
	const Tensor shuffle(uint32_t *pattern) const;
	/**
	 * @brief Changes shape of the Tensor.
	 * 
	 * @param new_shape New shape of the Tensor (size must remain same).
	 * @return Tensor after changing shape. 
	 */
	const Tensor reshape(std::vector<uint32_t> new_shape) const;

	/**
	 * @brief Prints Tensor as 'Tensor(<shape>).
	 * 
	 */
	void print() const;

private:
    /**
     * Shape of the tensor.
     */ 
	std::vector<uint32_t> _shape;
    /**
     * Size of the tensor (product of all shape elements).
     */ 
	uint32_t _size;
    /**
     * Values of the tensor.
     */ 
	std::vector<float> _data;

    /**
     * Validates shapes of Tensors starting from most outer.
	 * @param other Another Tensor object.
     * @return True if shapes are equal up to index (counting from beginning) which is a minimum of both tensors dimensions.
     */
	bool validateShape(const Tensor& other) const;
    /**
     * Validates shapes of Tensors starting from most inner.
	 * @param other Another Tensor object.
     * @return True if shapes are equal up to index (counting from end) which is a minimum of both tensors dimensions.
     */
	bool validateShapeReversed(const Tensor& other) const;

	friend class TensorSlice;
	friend class TensorCell;
};