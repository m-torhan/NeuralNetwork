#pragma once
#include <cstdint>

class Tensor {
public:
	Tensor(uint32_t dim, uint32_t* shape);
	Tensor(uint32_t dim ...);
	~Tensor();

	uint32_t* getShape();
	uint32_t getDim();
	float getValue(size_t i ...);
	void setValue(float value, size_t i ...);
	Tensor* operator+(Tensor* other);
	Tensor* operator-(Tensor* other);
	Tensor* operator*(Tensor* other);
	Tensor* operator/(Tensor* other);
	Tensor* operator+=(Tensor* other);
	Tensor* operator-=(Tensor* other);
	Tensor* operator*=(Tensor* other);
	Tensor* operator/=(Tensor* other);
	Tensor* operator+(float number);
	Tensor* operator-(float number);
	Tensor* operator*(float number);
	Tensor* operator/(float number);
	Tensor* operator+=(float number);
	Tensor* operator-=(float number);
	Tensor* operator*=(float number);
	Tensor* operator/=(float number);
	float dotProduct(Tensor* other);
	Tensor* tensorProduct(Tensor* other);

private:
	uint32_t _dim;
	uint32_t* _shape;
	uint32_t _size;
	float* _data;

	bool validateDimGreater(Tensor* other);
	bool validateDimEqual(Tensor* other);
	bool validateShape(Tensor* other);
};