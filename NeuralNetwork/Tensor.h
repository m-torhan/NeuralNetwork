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
	uint32_t getSize() const;
	float* getData() const;
	float getValue(size_t i ...) const;
	void setValue(float value, size_t i ...);
	Tensor* operator-() const;
	Tensor* operator+(const Tensor& other) const;
	Tensor* operator-(const Tensor& other) const;
	Tensor* operator*(const Tensor& other) const;
	Tensor* operator/(const Tensor& other) const;
	Tensor* operator+=(const Tensor& other);
	Tensor* operator-=(const Tensor& other);
	Tensor* operator*=(const Tensor& other);
	Tensor* operator/=(const Tensor& other);
	Tensor* operator>(const Tensor& other) const;
	Tensor* operator+(float number) const;
	Tensor* operator-(float number) const;
	Tensor* operator*(float number) const;
	Tensor* operator/(float number) const;
	Tensor* operator+=(float number);
	Tensor* operator-=(float number);
	Tensor* operator*=(float number);
	Tensor* operator/=(float number);
	Tensor* operator>(float other) const;
	Tensor* dotProduct(const Tensor& other) const;
	Tensor* tensorProduct(const Tensor& other) const;
	Tensor* applyFunction(float (*function)(float)) const;
	Tensor* flatten(uint32_t from_axis=0) const;
	Tensor* sum(uint32_t axis) const;
	float sum() const;
	Tensor* transpose() const;
	Tensor* slice(uint32_t axis, uint32_t start_idx, uint32_t end_idx) const;
	Tensor* shuffle() const;
	Tensor* shuffle(uint32_t *pattern) const;

private:
	uint32_t _dim;
	uint32_t* _shape;
	uint32_t _size;
	float* _data;

	bool validateDimGreater(const Tensor& other) const;
	bool validateDimEqual(const Tensor& other) const;
	bool validateShape(const Tensor& other) const;
};