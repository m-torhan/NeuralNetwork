#pragma once
#include <cstdint>

class Tensor {
public:
	Tensor(uint32_t dim, uint32_t* shape);
	Tensor(uint32_t dim ...);
	Tensor(const Tensor& other);
	Tensor();
	~Tensor();

	uint32_t* getShape() const;
	uint32_t getDim() const;
	float* getData() const;
	float getValue(size_t i ...) const;
	void setValue(float value, size_t i ...);
	Tensor* operator+(const Tensor& other);
	Tensor* operator-(const Tensor& other);
	Tensor* operator*(const Tensor& other);
	Tensor* operator/(const Tensor& other);
	Tensor* operator+=(const Tensor& other);
	Tensor* operator-=(const Tensor& other);
	Tensor* operator*=(const Tensor& other);
	Tensor* operator/=(const Tensor& other);
	Tensor* operator+(const float number);
	Tensor* operator-(const float number);
	Tensor* operator*(const float number);
	Tensor* operator/(const float number);
	Tensor* operator+=(const float number);
	Tensor* operator-=(const float number);
	Tensor* operator*=(const float number);
	Tensor* operator/=(const float number);
	Tensor* dotProduct(const Tensor& other);
	Tensor* tensorProduct(const Tensor& other);
	void applyFunction(float (*function)(float));

private:
	uint32_t _dim;
	uint32_t* _shape;
	uint32_t _size;
	float* _data;

	bool validateDimGreater(const Tensor& other);
	bool validateDimEqual(const Tensor& other);
	bool validateShape(const Tensor& other);
};