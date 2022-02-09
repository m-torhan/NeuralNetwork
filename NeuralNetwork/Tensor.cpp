#include "Tensor.h"

#include <cstdlib>
#include <cstring>
#include <cstdarg>

Tensor::Tensor(uint32_t dim, uint32_t* shape) {
	uint32_t size = 1;
	uint32_t i = 0;

	_dim = dim;
	_shape = (uint32_t*)malloc(sizeof(uint32_t) * dim);
	if (!_shape) {
		// exception
	}
	memcpy(_shape, shape, sizeof(uint32_t) * dim);

	for (i = 0; i < _dim; ++i) {
		size *= _shape[i];
	}
	_size = size;

	_data = (float*)malloc(sizeof(float) * _size);
}

Tensor::Tensor(uint32_t dim ...) {
	va_list shape;
	uint32_t size = 1;
	uint32_t i = 0;

	_dim = dim;
	_shape = (uint32_t*)malloc(sizeof(uint32_t) * dim);
	if (!_shape) {
		// exception
	}

	va_start(shape, dim);
	for (i = 0; i < _dim; ++i) {
		_shape[i] = va_arg(shape, uint32_t);
		size *= _shape[i];
	}
	_size = size;

	_data = (float*)malloc(sizeof(float) * _size);
}

Tensor::~Tensor() {
	free(_shape);
	free(_data);
}

uint32_t Tensor::getDim() {
	return _dim;
}

uint32_t* Tensor::getShape() {
	uint32_t* shape;

	shape = (uint32_t*)malloc(sizeof(uint32_t) * _dim);
	if (!shape) {
		// exception
	}
	memcpy(shape, _shape, sizeof(uint32_t) * _dim);

	return shape;
}

float Tensor::getValue(size_t n ...) {
	va_list indices;
	uint32_t idx = 0;
	uint32_t size_prod = this->_size;
	uint32_t i = 0;

	if (n != this->_dim) {
		// exception
	}

	va_start(indices, n);
	for (i = 0; i < this->_dim; ++i) {
		size_prod /= this->_shape[i];
		idx += size_prod * va_arg(indices, uint32_t);
	}

	return this->_data[idx];
}

void Tensor::setValue(float value, size_t n ...) {
	va_list indices;
	uint32_t idx = 0;
	uint32_t size_prod = this->_size;
	uint32_t i = 0;

	if (n != this->_dim) {
		// exception
	}

	va_start(indices, n);
	for (i = 0; i < this->_dim; ++i) {
		size_prod /= this->_shape[i];
		idx += size_prod * va_arg(indices, uint32_t);
	}

	this->_data[idx] = value;
}

Tensor* Tensor::operator+(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateShape(other)) {
		// exception
	}

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] + other->_data[i % other->_size];
	}

	return &result;
}

Tensor* Tensor::operator-(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] - other->_data[i % other->_size];
	}

	return &result;
}

Tensor* Tensor::operator*(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] * other->_data[i % other->_size];
	}

	return &result;
}

Tensor* Tensor::operator/(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] / other->_data[i % other->_size];
	}

	return &result;
}

Tensor* Tensor::operator+=(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] += other->_data[i % other->_size];
	}

	return this;
}

Tensor* Tensor::operator-=(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] -= other->_data[i % other->_size];
	}

	return this;
}

Tensor* Tensor::operator*=(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] *= other->_data[i % other->_size];
	}

	return this;
}

Tensor* Tensor::operator/=(Tensor* other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] /= other->_data[i % other->_size];
	}

	return this;
}

Tensor* Tensor::operator+(float number) {
	uint32_t i = 0;

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] + number;
	}

	return &result;
}

Tensor* Tensor::operator-(float number) {
	uint32_t i = 0;

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] - number;
	}

	return &result;
}

Tensor* Tensor::operator*(float number) {
	uint32_t i = 0;

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] * number;
	}

	return &result;
}

Tensor* Tensor::operator/(float number) {
	uint32_t i = 0;

	Tensor result = Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] / number;
	}

	return &result;
}

Tensor* Tensor::operator+=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] += number;
	}

	return this;
}

Tensor* Tensor::operator-=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] -= number;
	}

	return this;
}

Tensor* Tensor::operator*=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] *= number;
	}

	return this;
}

Tensor* Tensor::operator/=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] /= number;
	}

	return this;
}

float Tensor::dotProduct(Tensor* other) {
	uint32_t i = 0;
	float sum = 0;

	if (!this->validateDimEqual(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		sum += this->_data[i] * other->_data[i];
	}

	return sum;
}

Tensor* Tensor::tensorProduct(Tensor* other) {
	Tensor ret = Tensor(2, 3, 3);
	return &ret;
}

bool Tensor::validateDimGreater(Tensor* other) {
	return this->_dim >= other->_dim;
}

bool Tensor::validateDimEqual(Tensor* other) {
	return this->_dim == other->_dim;
}

bool Tensor::validateShape(Tensor* other) {
	uint32_t i = 0;
	uint32_t min_dim = 0;

	if (this->_dim < other->_dim) {
		min_dim = this->_dim;
	}
	else {
		min_dim = other->_dim;
	}
	
	for (i = 0; i < min_dim; ++i) {
		if (this->_shape[i] != other->_shape[i]) {
			return false;
		}
	}

	return true;
}