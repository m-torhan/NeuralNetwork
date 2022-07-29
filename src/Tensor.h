#pragma once

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <algorithm>
#include <cstdio>

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

class TensorSlice {
public:
	const Tensor& operator=(const Tensor& other);

private:
	TensorSlice();
	TensorSlice(Tensor& tensor, std::vector<std::vector<uint32_t> > slice_ranges) :
		_tensor(tensor), _slice_ranges(slice_ranges) {}

	Tensor& _tensor;
	std::vector<std::vector<uint32_t> > _slice_ranges;
	
	friend class Tensor;
};

class TensorCell {
public:
	float operator=(float value);

private:
	TensorCell();
	TensorCell(Tensor& tensor, std::vector<uint32_t> cell_index) :
		_tensor(tensor), _cell_index(cell_index) {}

	Tensor& _tensor;
	std::vector<uint32_t> _cell_index;

	friend class Tensor;
};

class Tensor {
public:
	Tensor(const std::vector<uint32_t>& shape);
	Tensor(const Tensor& other);
	const Tensor& operator=(const Tensor& other);
	const Tensor& operator=(const Tensor&& other);
	Tensor();
	~Tensor();

	const Tensor operator[](const std::vector<std::vector<uint32_t> >& ranges) const;
	TensorSlice operator[](const std::vector<std::vector<uint32_t> >& ranges);
	const float operator[](const std::vector<uint32_t>& index) const;
	const float operator[](const std::initializer_list<uint32_t>& ini) const
	{
		return operator[](std::vector<uint32_t>(ini));
	}
	TensorCell operator[](const std::vector<uint32_t>& index);
	TensorCell operator[](const std::initializer_list<uint32_t>& ini)
	{
		return operator[](std::vector<uint32_t>(ini));
	}

	std::vector<uint32_t> getShape() const;
	uint32_t getDim() const;
	uint32_t getSize() const;
	std::vector<float> getData() const;

	void setValues(const std::vector<float>& values);

	const Tensor operator-() const;

	Tensor& operator+=(const Tensor& other);
	const Tensor operator+(const Tensor& other) const;
	Tensor& operator+=(float number);
	const Tensor operator+(float number) const;
	friend const Tensor operator+(float number, const Tensor& other);

	Tensor& operator-=(const Tensor& other);
	const Tensor operator-(const Tensor& other) const;
	Tensor& operator-=(float number);
	const Tensor operator-(float number) const;
	friend const Tensor operator-(float number, const Tensor& other);

	Tensor& operator*=(const Tensor& other);
	const Tensor operator*(const Tensor& other) const;
	Tensor& operator*=(float number);
	const Tensor operator*(float number) const;
	friend const Tensor operator*(float number, const Tensor& other);

	Tensor& operator/=(const Tensor& other);
	const Tensor operator/(const Tensor& other) const;
	Tensor& operator/=(float number);
	const Tensor operator/(float number) const;
	friend const Tensor operator/(float number, const Tensor& other);

	const Tensor operator==(const Tensor& other) const;
	const Tensor operator!=(const Tensor& other) const;
	const Tensor operator>(const Tensor& other) const;
	const Tensor operator>=(const Tensor& other) const;
	const Tensor operator<(const Tensor& other) const;
	const Tensor operator<=(const Tensor& other) const;
	
	const Tensor operator==(float other) const;
	const Tensor operator!=(float other) const;
	const Tensor operator>(float other) const;
	const Tensor operator>=(float other) const;
	const Tensor operator<(float other) const;
	const Tensor operator<=(float other) const;
	
	const Tensor addPadding(const std::vector<uint32_t>& axes, const std::vector<Padding>& paddings, const std::vector<uint32_t>& counts) const;
	const Tensor dotProduct(const Tensor& other) const;
	const Tensor dotProductTranspose(const Tensor& other) const;
	const Tensor tensorProduct(const Tensor& other) const;
	const Tensor applyFunction(float (*function)(float)) const;
	const Tensor flatten(uint32_t from_axis=0) const;
	const Tensor conv2D(const Tensor& other) const;
	const Tensor sum(uint32_t axis) const;
	float sum() const;
	float max() const;
	float mean() const;
	const Tensor transpose() const;
	const Tensor slice(uint32_t axis, uint32_t start_idx, uint32_t end_idx) const;
	const Tensor shuffle() const;
	const Tensor shuffle(uint32_t *pattern) const;
	const Tensor reshape(std::vector<uint32_t> new_shape) const;

	void print() const;

private:
	std::vector<uint32_t> _shape;
	uint32_t _size;
	std::vector<float> _data;

	bool validateShape(const Tensor& other) const;
	bool validateShapeReversed(const Tensor& other) const;

	friend class TensorSlice;
	friend class TensorCell;
};