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
		void SSE_tensor_add_scalar(const uint32_t n1, const float* v, const float* s, float* r);
		
		void SSE_vector_sub(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_sub(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_sub_scalar(const uint32_t n, const float* v, const float* s, float* r);
		void SSE_scalar_sub_tensor(const float* s, const uint32_t n, const float* v, float* r);

		void SSE_vector_mul(const uint32_t n, const float* v1, const float* v2, float* r);
		void SSE_tensor_mul(const uint32_t n1, const float* v1, const uint32_t n2, const float* v2, float* r);
		void SSE_tensor_mul_scalar(const uint32_t n1, const float* v, const float* s, float* r);

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


class Tensor {
public:
	Tensor(const std::vector<uint32_t>& shape);
	Tensor(const Tensor& other);
	Tensor& operator=(const Tensor& other);
	Tensor();
	~Tensor();

	std::vector<uint32_t> getShape() const;
	uint32_t getDim() const;
	uint32_t getSize() const;
	std::vector<float> getData() const;
	float getValue(const std::vector<uint32_t>& idx = { 0 }) const;
	void setValue(float value, const std::vector<uint32_t>& idx = { 0 });
	void setValues(	const std::vector<float>& values);
	const Tensor operator-() const;
	const Tensor operator+(const Tensor& other) const;
	const Tensor operator-(const Tensor& other) const;
	const Tensor operator*(const Tensor& other) const;
	const Tensor operator/(const Tensor& other) const;
	Tensor& operator+=(const Tensor& other);
	Tensor& operator-=(const Tensor& other);
	Tensor& operator*=(const Tensor& other);
	Tensor& operator/=(const Tensor& other);
	const Tensor operator>(const Tensor& other) const;
	const Tensor operator<(const Tensor& other) const;
	const Tensor operator+(float number) const;
	friend const Tensor operator+(float number, const Tensor& other);
	const Tensor operator-(float number) const;
	friend const Tensor operator-(float number, const Tensor& other);
	const Tensor operator*(float number) const;
	friend const Tensor operator*(float number, const Tensor& other);
	const Tensor operator/(float number) const;
	friend const Tensor operator/(float number, const Tensor& other);
	Tensor& operator+=(float number);
	Tensor& operator-=(float number);
	Tensor& operator*=(float number);
	Tensor& operator/=(float number);
	const Tensor operator>(float other) const;
	const Tensor operator<(float other) const;
	const Tensor dotProduct(const Tensor& other) const;
	const Tensor dotProductTranspose(const Tensor& other) const;
	const Tensor tensorProduct(const Tensor& other) const;
	const Tensor applyFunction(float (*function)(float)) const;
	const Tensor flatten(uint32_t from_axis=0) const;
	const Tensor sum(uint32_t axis) const;
	float sum() const;
	const Tensor transpose() const;
	const Tensor slice(uint32_t axis, uint32_t start_idx, uint32_t end_idx) const;
	const Tensor shuffle() const;
	const Tensor shuffle(uint32_t *pattern) const;

private:
	std::vector<uint32_t> _shape;
	uint32_t _size;
	std::vector<float> _data;

	bool validateDimGreater(const Tensor& other) const;
	bool validateDimEqual(const Tensor& other) const;
	bool validateShape(const Tensor& other) const;
	bool validateShapeReversed(const Tensor& other) const;
};