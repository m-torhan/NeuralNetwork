#pragma once

#include <cstdint>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>
#include <algorithm>

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
	const Tensor operator-(float number) const;
	const Tensor operator*(float number) const;
	const Tensor operator/(float number) const;
	Tensor& operator+=(float number);
	Tensor& operator-=(float number);
	Tensor& operator*=(float number);
	Tensor& operator/=(float number);
	const Tensor operator>(float other) const;
	const Tensor operator<(float other) const;
	const Tensor dotProduct(const Tensor& other) const;
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
};