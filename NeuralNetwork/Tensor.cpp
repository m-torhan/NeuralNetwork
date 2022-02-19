#include "Tensor.h"

#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <ctime>

Tensor::Tensor(uint32_t dim, uint32_t* shape) {
	uint32_t size = 1;
	uint32_t i = 0;

	_dim = dim;
	_shape = (uint32_t*)malloc(sizeof(uint32_t) * _dim);
	if (!_shape) {
		// exception
	}
	memcpy(_shape, shape, sizeof(uint32_t) * _dim);

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
	_shape = (uint32_t*)malloc(sizeof(uint32_t) * _dim);
	if (!_shape) {
		// exception
	}

	va_start(shape, dim);
	for (i = 0; i < _dim; ++i) {
		_shape[i] = va_arg(shape, uint32_t);
		size *= _shape[i];
	}
	va_end(shape);

	_size = size;

	_data = (float*)malloc(sizeof(float) * _size);
}

Tensor::Tensor(const Tensor& other) {
	_dim = other._dim;
	_size = other._size;

	_shape = (uint32_t*)malloc(sizeof(uint32_t) * _dim);
	if (!_shape) {
		// exception
	}
	memcpy(_shape, other._shape, sizeof(uint32_t) * _dim);

	_data = (float*)malloc(sizeof(float) * _size);
	if (!_data) {
		// exception
	}
	memcpy(_data, other._data, sizeof(float) * _size);
}

Tensor::Tensor() {
	_dim = 0;
	_shape = 0;
	_size = 0;
	_data = 0;
}

Tensor::~Tensor() {
	free(_shape);
	free(_data);
}

uint32_t Tensor::getDim() const {
	return _dim;
}

uint32_t Tensor::getSize() const {
	return _size;
}

uint32_t* Tensor::getShape() const {
	uint32_t* shape;

	shape = (uint32_t*)malloc(sizeof(uint32_t) * _dim);
	if (!shape) {
		// exception
	}
	memcpy(shape, _shape, sizeof(uint32_t) * _dim);

	return shape;
}

float* Tensor::getData() const {
	float* data;

	data = (float*)malloc(sizeof(float) * _size);
	if (!data) {
		// exception
	}
	memcpy(data, _data, sizeof(float) * _size);

	return data;
}

float Tensor::getValue(size_t n ...) const {
	va_list indices;
	uint32_t idx = 0;
	uint32_t subidx = 0;
	uint32_t subsize = this->_size;
	uint32_t i = 0;

	if (n != this->_dim) {
		// exception
	}

	va_start(indices, n);
	for (i = 0; i < this->_dim; ++i) {
		subsize /= this->_shape[i];
		subidx = va_arg(indices, uint32_t);
		if (subidx >= this->_shape[i]) {
			// exception
		}
		idx += subsize * subidx;
	}
	va_end(indices);

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
	va_end(indices);

	this->_data[idx] = value;
}

Tensor* Tensor::operator-() const {
	return *this * (-1.0f);
}

Tensor* Tensor::operator+(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateShape(other)) {
		// exception
	}

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] + other._data[i % other._size];
	}

	return result;
}

Tensor* Tensor::operator-(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] - other._data[i % other._size];
	}

	return result;
}

Tensor* Tensor::operator*(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] * other._data[i % other._size];
	}

	return result;
}

Tensor* Tensor::operator/(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] / other._data[i % other._size];
	}

	return result;
}

Tensor* Tensor::operator+=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] += other._data[i % other._size];
	}

	return this;
}

Tensor* Tensor::operator-=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] -= other._data[i % other._size];
	}

	return this;
}

Tensor* Tensor::operator*=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] *= other._data[i % other._size];
	}

	return this;
}

Tensor* Tensor::operator/=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] /= other._data[i % other._size];
	}

	return this;
}

Tensor* Tensor::operator>(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] > other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

Tensor* Tensor::operator+(float number) const {
	uint32_t i = 0;

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] + number;
	}

	return result;
}

Tensor* Tensor::operator-(float number) const {
	uint32_t i = 0;

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] - number;
	}

	return result;
}

Tensor* Tensor::operator*(float number) const {
	uint32_t i = 0;

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] * number;
	}

	return result;
}

Tensor* Tensor::operator/(float number) const {
	uint32_t i = 0;

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] / number;
	}

	return result;
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

Tensor* Tensor::operator>(float number) const {
	uint32_t i = 0;

	Tensor* result = new Tensor(this->_dim, this->_shape);

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = this->_data[i] > number ? 1.0f : 0.0f;
	}

	return result;
}

Tensor* Tensor::dotProduct(const Tensor& other) const {
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t k = 0;
	uint32_t d_i;
	uint32_t d_j;
	uint32_t result_dim;
	uint32_t* result_shape;
	Tensor* result;

	if (this->_dim == 1 && other._dim == 1) {
		// vector inner product
		if (this->_shape[0] != other._shape[0]) {
			// exception
		}
		result = new Tensor(0);

		*result *= 0.0f;

		for (i = 0; i < this->_size; ++i) {
			result->_data[0] += this->_data[i] * other._data[i];
		}

		return result;
	}

	if (this->_dim == 2 && other._dim == 2) {
		// matrix multiplication
		if (this->_shape[1] != other._shape[0]) {
			// exception
		}
		result_dim = 2;
		result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);
		if (!result_shape) {
			// exception
		}
		result_shape[0] = this->_shape[0];
		result_shape[1] = other._shape[1];

		result = new Tensor(result_dim, result_shape);

		*result *= 0.0f;

		for (i = 0; i < result_shape[0]; ++i) {
			for (j = 0; j < result_shape[1]; ++j) {
				for (k = 0; k < this->_shape[1]; ++k) {
					result->_data[i * result_shape[1] + j] += this->_data[i * this->_shape[1] + k] * other._data[k * other._shape[1] + j];
				}
			}
		}

		free(result_shape);

		return result;
	}

	if (this->_dim == 0) {
		// scalar multiplication
		return other * this->_data[0];
	}

	if (other._dim == 0) {
		// scalar multiplication
		return *this * other._data[0];
	}

	if (other._dim == 1) {
		// sum product over the last axis of this and other (vector)
		if (this->_shape[this->_dim - 1] != other._shape[0]) {
			// exception
		}
		result_dim = this->_dim - 1;
		result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);

		memcpy(result_shape, this->_shape, sizeof(uint32_t) * result_dim);

		result = new Tensor(result_dim, result_shape);

		*result *= 0.0f;

		d_i = this->_size / other._shape[0];

		for (i = 0; i < other._shape[0]; ++i) {
			for (j = 0; j < d_i; ++j) {
				result->_data[j] += this->_data[i * d_i + j] * other._data[i];
			}
		}

		free(result_shape);

		return result;
	}

	else {
		// sum product over the last axis of this and the second-to-last axis of other
		if (this->_shape[this->_dim - 1] != other._shape[other._dim - 2]) {
			// exception
		}
		result_dim = this->_dim + other._dim - 2;
		result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);

		result = new Tensor(result_dim, result_shape);

		*result *= 0.0f;

		d_i = this->_size / this->_shape[this->_dim - 1];
		d_j = other._size / other._shape[other._dim - 2];

		// TODO (dot product for higher dimensions currently not essential)

		free(result_shape);

		return result;
	}
}

Tensor* Tensor::tensorProduct(const Tensor& other) const {
	uint32_t i = 0;
	Tensor* result;
	Tensor* subresult;
	uint32_t result_dim = this->_dim + other._dim;
	uint32_t* result_shape;

	result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);
	if (!result_shape) {
		// exception
	}
	memcpy(result_shape, this->_shape, sizeof(uint32_t) * this->_dim);
	memcpy(&result_shape[this->_dim], other._shape, sizeof(uint32_t) * other._dim);

	result = new Tensor(result_dim, result_shape);

	for (i = 0; i < this->_size; ++i) {
		subresult = new Tensor(other);
		*subresult *= this->_data[i];
		memcpy(&(result->_data[subresult->_size * i]), subresult->_data, sizeof(float) * subresult->_size);
	}

	free(result_shape);

	return result;
}

Tensor* Tensor::applyFunction(float (*function)(float)) const {
	Tensor* result = new Tensor(*this);
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		result->_data[i] = function(result->_data[i]);
	}

	return result;
}

Tensor* Tensor::flatten(uint32_t from_axis) const {
	uint32_t i = 0;
	uint32_t subsize = 1;
	uint32_t result_dim;
	uint32_t* result_shape;
	Tensor* result;

	if (from_axis >= this->_dim) {
		// exception
	}

	for (i = from_axis; i < this->_dim; ++i) {
		subsize *= this->_shape[i];
	}

	result_dim = 1 + from_axis;

	result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);
	if (result_shape) {
		// exception
	}

	for (i = 0; i < from_axis; ++i) {
		result_shape[i] = this->_shape[i];
	}
	result_shape[from_axis] = subsize;

	result = new Tensor(result_dim, result_shape);

	memcpy(result->_data, this->_data, sizeof(float) * this->_dim);

	free(result_shape);

	return result;
}

Tensor* Tensor::sum(uint32_t axis) const {
	Tensor* result;
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t k = 0;
	uint32_t d_i = 0;
	uint32_t d_j = 0;
	uint32_t d_k = 0;
	uint32_t result_dim;
	uint32_t* result_shape;

	if (axis >= this->_dim) {
		// exception
	}

	result_dim = this->_dim - 1;
	result_shape = (uint32_t*)malloc(sizeof(uint32_t) * result_dim);
	if (!result_shape) {
		// exception
	}
	for (i = 0; i < this->_dim; ++i) {
		if (i != axis) {
			result_shape[i - (i > axis ? 1 : 0)] = this->_shape[i];
		}
	}

	result = new Tensor(result_dim, result_shape);

	d_j = axis == this->_dim - 1 ? this->_shape[this->_dim - 1] : 1;

	d_i = 1;
	for (i = axis + 1; i < this->_dim; ++i) {
		d_i *= this->_shape[i];
	}
	d_k = d_i * this->_shape[axis];

	*result *= 0.0f;

	for (k = 0; k < this->_size / d_k; ++k) {
		for (j = 0; j < d_i; ++j) {
			for (i = 0; i < this->_shape[axis] ; ++i) {
				result->_data[j + d_i * k] += this->_data[d_i * i + d_j * j + d_k * k];
			}
		}
	}

	free(result_shape);
	return result;
}

float Tensor::sum() const {
	uint32_t i{ 0 };
	float result{ 0.0f };

	for (i = 0; i < this->_size; ++i) {
		result += this->_data[i];
	}

	return result;
}

Tensor* Tensor::transpose() const {
	Tensor* result;
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t* result_shape;

	if (this->_dim != 2) {
		// exception
	}

	result_shape = (uint32_t*)malloc(sizeof(uint32_t) * this->_dim);
	if (!result_shape) {
		// exception
	}

	result_shape[0] = this->_shape[1];
	result_shape[1] = this->_shape[0];

	result = new Tensor(this->_dim, result_shape);

	for (i = 0; i < result_shape[0]; ++i) {
		for (j = 0; j < result_shape[1]; ++j) {
			result->_data[i * result_shape[1] + j] = this->_data[j * result_shape[0] + i];
		}
	}

	free(result_shape);

	return result;
}

Tensor* Tensor::slice(uint32_t axis, uint32_t start_idx, uint32_t end_idx) const {
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t subsize_1 = 1;
	uint32_t subsize_2 = 1;
	Tensor* result;
	uint32_t* result_shape;

	if (start_idx <= end_idx || axis >= this->_dim) {
		// exception
	}

	result_shape = (uint32_t*)malloc(sizeof(uint32_t) * this->_dim);
	if (!result_shape) {
		// exception
	}
	memcpy(result_shape, this->_shape, sizeof(uint32_t) * this->_dim);
	result_shape[axis] = end_idx - start_idx;

	result = new Tensor(this->_dim, result_shape);

	for (i = axis + 1; i < this->_dim; ++i) {
		subsize_2 *= this->_shape[i];
	}
	subsize_1 = this->_shape[axis] * subsize_2;

	for (i = 0; i < this->_size; ++i) {
		if (start_idx * subsize_2 <= i % subsize_1 && i % subsize_1 < end_idx * subsize_2) {
			result->_data[j] = this->_data[i];
			++j;
		}
	}

	free(result_shape);

	return result;
}

Tensor* Tensor::shuffle() const {
	uint32_t axis = 0; // currently only for first axis
	uint32_t i = 0;
	uint32_t rand_a;
	uint32_t rand_b;
	Tensor* result;
	uint32_t subsize = 0;
	uint32_t shuffle_count = 0;
	float* tmp;

	result = new Tensor(*this);

	subsize = this->_size / this->_shape[axis];
	tmp = (float*)malloc(sizeof(float) * subsize);
	if (!tmp) {
		// exception
	}

	srand(time(NULL));

	shuffle_count = this->_shape[axis] + rand() % this->_shape[axis];

	for (i = 0; i < shuffle_count; ++i) {
		rand_a = rand() % this->_shape[axis];
		rand_b = rand() % this->_shape[axis];

		memcpy(tmp, &result->_data[subsize * rand_a], sizeof(float) * subsize);
		memcpy(&result->_data[subsize * rand_a], &result->_data[subsize * rand_b], sizeof(float) * subsize);
		memcpy(&result->_data[subsize * rand_b], tmp, sizeof(float) * subsize);
	}

	free(tmp);

	return result;
}

Tensor* Tensor::shuffle(uint32_t *pattern) const {
	uint32_t axis = 0; // currently only for first axis
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t* shuffled;
	Tensor* result;
	uint32_t subsize = 0;
	float* tmp;

	result = new Tensor(*this);

	subsize = this->_size / this->_shape[axis];
	tmp = (float*)malloc(sizeof(float) * subsize);
	if (!tmp) {
		// exception
	}
	shuffled = (uint32_t*)malloc(sizeof(uint32_t) * this->_shape[axis]);
	if (!shuffled) {
		// exception
	}
	memset(shuffled, 0, sizeof(uint32_t) * this->_shape[axis]);

	for (i = 0; i < this->_shape[axis]; ++i) {
		if (!shuffled[i]) {
			shuffled[i] = 1;
			j = pattern[i];
			do {
				shuffled[j] = 1;

				memcpy(tmp, &result->_data[subsize * j], sizeof(float) * subsize);
				memcpy(&result->_data[subsize * j], &result->_data[subsize * pattern[j]], sizeof(float) * subsize);
				memcpy(&result->_data[subsize * pattern[j]], tmp, sizeof(float) * subsize);
			
				j = pattern[j];
			} while (j != i);
		}
	}

	free(tmp);
	free(shuffled);

	return result;
}

bool Tensor::validateDimGreater(const Tensor& other) const {
	return this->_dim >= other._dim;
}

bool Tensor::validateDimEqual(const Tensor& other) const {
	return this->_dim == other._dim;
}

bool Tensor::validateShape(const Tensor& other) const {
	uint32_t i = 0;
	uint32_t min_dim = 0;

	if (this->_dim < other._dim) {
		min_dim = this->_dim;
	}
	else {
		min_dim = other._dim;
	}
	
	for (i = 0; i < min_dim; ++i) {
		if (this->_shape[i] != other._shape[i]) {
			return false;
		}
	}

	return true;
}