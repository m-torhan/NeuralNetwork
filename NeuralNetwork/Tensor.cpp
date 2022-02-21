#include "Tensor.h"

Tensor::Tensor(const std::vector<uint32_t>& shape) {
	_shape = shape;

	_size = 1;
	for (auto s : shape) {
		_size *= s;
	}

	_data.insert(_data.end(), _size, 0.0f);
}

Tensor::Tensor(const Tensor& other) {
	_size = other._size;
	_shape = other._shape;
	_data = other._data;
}

Tensor& Tensor::operator=(const Tensor& other) {
	_size = other._size;
	_shape = other._shape;
	_data = other._data;

	return *this;
}

Tensor::Tensor() {
	_size = 1;
	_shape.push_back(1);
	_data.push_back(0.0f);
}

Tensor::~Tensor() {
}

uint32_t Tensor::getDim() const {
	return _shape.size();
}

uint32_t Tensor::getSize() const {
	return _size;
}

std::vector<uint32_t> Tensor::getShape() const {
	return _shape;
}

std::vector<float> Tensor::getData() const {
	return _data;
}

float Tensor::getValue(const std::vector<uint32_t>& idx) const {
	uint32_t flat_idx = 0;
	uint32_t subidx = 0;
	uint32_t subsize = this->_size;
	uint32_t i = 0;

	if (idx.size() != this->_shape.size()) {
		// exception
	}

	for (i = 0; i < this->_shape.size(); ++i) {
		subsize /= this->_shape[i];
		subidx = idx[i];
		if (subidx >= this->_shape[i]) {
			// exception
		}
		flat_idx += subsize * subidx;
	}

	return this->_data[flat_idx];
}

void Tensor::setValue(float value, const std::vector<uint32_t>& idx) {
	if (idx.size() != this->_shape.size()) {
		// exception
	}

	uint32_t flat_idx = 0;
	uint32_t subsize = this->_size;
	for (uint32_t i = 0; i < this->_shape.size(); ++i) {
		subsize /= this->_shape[i];
		flat_idx += subsize * idx[i];
	}

	this->_data[flat_idx] = value;
}

void Tensor::setValues(const std::vector<float>& values) {
	if (this->_data.size() != values.size()) {
		// exception
	}
	this->_data = values;
}

const Tensor Tensor::operator-() const {
	Tensor result = *this;
	return result * (-1.0f);
}

const Tensor Tensor::operator+(const Tensor& other) const {
	Tensor result = *this;
	result += other;
	return result;
}

const Tensor Tensor::operator-(const Tensor& other) const {
	Tensor result = *this;
	result -= other;
	return result;
}

const Tensor Tensor::operator*(const Tensor& other) const {
	Tensor result = *this;
	result *= other;
	return result;
}

const Tensor Tensor::operator/(const Tensor& other) const {
	Tensor result = *this;
	result /= other;
	return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] += other._data[i % other._size];
	}

	return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] -= other._data[i % other._size];
	}

	return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] *= other._data[i % other._size];
	}

	return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] /= other._data[i % other._size];
	}

	return *this;
}

const Tensor Tensor::operator>(const Tensor& other) const {
	uint32_t i = 0;

	if (!this->validateDimGreater(other) ||
		!this->validateShape(other)) {
		// exception
	}

	Tensor result = *this;

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] > other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator+(float number) const {
	Tensor result = *this;
	result += number;
	return result;
}

const Tensor Tensor::operator-(float number) const {
	Tensor result = *this;
	result -= number;
	return result;
}

const Tensor Tensor::operator*(float number) const {
	Tensor result = *this;
	result *= number;
	return result;
}

const Tensor Tensor::operator/(float number) const {
	Tensor result = *this;
	result /= number;
	return result;
}

Tensor& Tensor::operator+=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] += number;
	}

	return *this;
}

Tensor& Tensor::operator-=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] -= number;
	}

	return *this;
}

Tensor& Tensor::operator*=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] *= number;
	}

	return *this;
}

Tensor& Tensor::operator/=(float number) {
	uint32_t i = 0;

	for (i = 0; i < this->_size; ++i) {
		this->_data[i] /= number;
	}

	return *this;
}

const Tensor Tensor::operator>(float number) const {
	uint32_t i = 0;

	Tensor result = *this;

	for (i = 0; i < this->_size; ++i) {
		result._data[i] = this->_data[i] > number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::dotProduct(const Tensor& other) const {
	if (this->_shape.size() == 1 && other._shape.size() == 1) {
		// vector inner product
		if (this->_shape[0] != other._shape[0]) {
			// exception
		}
		Tensor result = Tensor();

		result *= 0.0f;

		for (uint32_t i = 0; i < this->_size; ++i) {
			result._data[0] += this->_data[i] * other._data[i];
		}

		return result;
	}

	if (this->_shape.size() == 2 && other._shape.size() == 2) {
		// matrix multiplication
		if (this->_shape[1] != other._shape[0]) {
			// exception
		}
		std::vector<uint32_t> result_shape = { this->_shape[0], other._shape[1] };

		Tensor result = Tensor(result_shape);

		for (uint32_t i = 0; i < result_shape[0]; ++i) {
			for (uint32_t j = 0; j < result_shape[1]; ++j) {
				for (uint32_t k = 0; k < this->_shape[1]; ++k) {
					result._data[i * result_shape[1] + j] += this->_data[i * this->_shape[1] + k] * other._data[k * other._shape[1] + j];
				}
			}
		}

		return result;
	}

	if (this->_shape.size() == 0) {
		// scalar multiplication
		return other * this->_data[0];
	}

	if (other._shape.size() == 0) {
		// scalar multiplication
		return *this * other._data[0];
	}

	if (other._shape.size() == 1) {
		// sum product over the last axis of this and other (vector)
		if (this->_shape[this->_shape.size() - 1] != other._shape[0]) {
			// exception
		}
		std::vector<uint32_t> result_shape = this->_shape;
		result_shape.pop_back();

		Tensor result = Tensor(result_shape);

		uint32_t d_i = this->_size / other._shape[0];

		for (uint32_t i = 0; i < other._shape[0]; ++i) {
			for (uint32_t j = 0; j < d_i; ++j) {
				result._data[j] += this->_data[i * d_i + j] * other._data[i];
			}
		}

		return result;
	}

	else {
		// sum product over the last axis of this and the second-to-last axis of other
		if (this->_shape[this->_shape.size()  - 1] != other._shape[other._shape.size()  - 2]) {
			// exception
		}

		Tensor result = Tensor();

		//d_i = this->_size / this->_shape[this->_shape.size()  - 1];
		//d_j = other._size / other._shape[other._shape.size()  - 2];

		// TODO (dot product for higher dimensions currently not essential)

		return result;
	}
}

const Tensor Tensor::tensorProduct(const Tensor& other) const {
	std::vector<uint32_t> result_shape = this->_shape;
	result_shape.insert(result_shape.end(), other._shape.begin(), other._shape.end());

	Tensor result = Tensor(result_shape);

	for (uint32_t i = 0; i < this->_size; ++i) {
		Tensor subresult = Tensor(other);
		subresult *= this->_data[i];
		std::copy(subresult._data.begin(), subresult._data.end(), result._data.begin() + i * subresult._data.size());
	}

	return result;
}

const Tensor Tensor::applyFunction(float (*function)(float)) const {
	Tensor result = *this;

	for (uint32_t i = 0; i < this->_size; ++i) {
		result._data[i] = function(result._data[i]);
	}

	return result;
}

const Tensor Tensor::flatten(uint32_t from_axis) const {
	if (from_axis >= this->_shape.size() ) {
		// exception
	}

	uint32_t subsize = 1;
	for (uint32_t i = from_axis; i < this->_shape.size() ; ++i) {
		subsize *= this->_shape[i];
	}

	std::vector<uint32_t> result_shape;

	for (uint32_t i = 0; i < from_axis; ++i) {
		result_shape.push_back(this->_shape[i]);
	}
	result_shape.push_back(subsize);

	Tensor result = Tensor(result_shape);

	std::copy(this->_data.begin(), this->_data.end(), result._data.begin());

	return result;
}

const Tensor Tensor::sum(uint32_t axis) const {
	if (axis >= this->_shape.size() ) {
		// exception
	}

	std::vector<uint32_t> result_shape;

	for (uint32_t i = 0; i < this->_shape.size() ; ++i) {
		if (i != axis) {
			result_shape.push_back(this->_shape[i]);
		}
	}

	Tensor result = Tensor(result_shape);

	uint32_t d_j = axis == this->_shape.size()  - 1 ? this->_shape[this->_shape.size()  - 1] : 1;

	uint32_t d_i = 1;
	for (uint32_t i = axis + 1; i < this->_shape.size() ; ++i) {
		d_i *= this->_shape[i];
	}
	uint32_t d_k = d_i * this->_shape[axis];

	for (uint32_t k = 0; k < this->_size / d_k; ++k) {
		for (uint32_t j = 0; j < d_i; ++j) {
			for (uint32_t i = 0; i < this->_shape[axis] ; ++i) {
				result._data[j + d_i * k] += this->_data[d_i * i + d_j * j + d_k * k];
			}
		}
	}

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

const Tensor Tensor::transpose() const {
	if (this->_shape.size()  != 2) {
		// exception
	}

	std::vector<uint32_t> result_shape = { this->_shape[1], this->_shape[0] };

	Tensor result = Tensor(result_shape);

	for (uint32_t i = 0; i < result_shape[0]; ++i) {
		for (uint32_t j = 0; j < result_shape[1]; ++j) {
			result._data[i * result_shape[1] + j] = this->_data[j * result_shape[0] + i];
		}
	}

	return result;
}

const Tensor Tensor::slice(uint32_t axis, uint32_t start_idx, uint32_t end_idx) const {
	if (start_idx <= end_idx || axis >= this->_shape.size() ) {
		// exception
	}

	std::vector<uint32_t> result_shape = this->_shape;
	result_shape[axis] = end_idx - start_idx;

	Tensor result = Tensor(result_shape);

	uint32_t subsize_2 = 1;
	for (uint32_t i = axis + 1; i < this->_shape.size() ; ++i) {
		subsize_2 *= this->_shape[i];
	}
	uint32_t subsize_1 = this->_shape[axis] * subsize_2;

	uint32_t j = 0;
	for (uint32_t i = 0; i < this->_size; ++i) {
		if (start_idx * subsize_2 <= i % subsize_1 && i % subsize_1 < end_idx * subsize_2) {
			result._data[j] = this->_data[i];
			++j;
		}
	}
	return result;
}

const Tensor Tensor::shuffle() const {
	uint32_t axis = 0; // currently only for first axis
	uint32_t i = 0;
	uint32_t rand_a;
	uint32_t rand_b;
	Tensor result;
	uint32_t subsize = 0;
	uint32_t shuffle_count = 0;
	float* tmp;

	result = *this;

	subsize = this->_size / this->_shape[axis];
	tmp = (float*)malloc(sizeof(float) * subsize);
	if (!tmp) {
		// exception
	}

	srand(time(NULL));

	shuffle_count = 2 * this->_shape[axis] + rand() % this->_shape[axis];

	for (i = 0; i < shuffle_count; ++i) {
		rand_a = rand() % this->_shape[axis];
		rand_b = rand() % this->_shape[axis];

		memcpy(tmp, &result._data[subsize * rand_a], sizeof(float) * subsize);
		memcpy(&result._data[subsize * rand_a], &result._data[subsize * rand_b], sizeof(float) * subsize);
		memcpy(&result._data[subsize * rand_b], tmp, sizeof(float) * subsize);
	}

	free(tmp);

	return result;
}

const Tensor Tensor::shuffle(uint32_t *pattern) const {
	uint32_t axis = 0; // currently only for first axis
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t* shuffled;
	Tensor result;
	uint32_t subsize = 0;
	float* tmp;

	result = *this;

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

				memcpy(tmp, &result._data[subsize * j], sizeof(float) * subsize);
				memcpy(&result._data[subsize * j], &result._data[subsize * pattern[j]], sizeof(float) * subsize);
				memcpy(&result._data[subsize * pattern[j]], tmp, sizeof(float) * subsize);
			
				j = pattern[j];
			} while (j != i);
		}
	}

	free(tmp);
	free(shuffled);

	return result;
}

bool Tensor::validateDimGreater(const Tensor& other) const {
	return this->_shape.size()  >= other._shape.size() ;
}

bool Tensor::validateDimEqual(const Tensor& other) const {
	return this->_shape.size()  == other._shape.size() ;
}

bool Tensor::validateShape(const Tensor& other) const {
	uint32_t i = 0;
	uint32_t min_dim = 0;

	if (this->_shape.size()  < other._shape.size() ) {
		min_dim = this->_shape.size() ;
	}
	else {
		min_dim = other._shape.size() ;
	}
	
	for (i = 0; i < min_dim; ++i) {
		if (this->_shape[i] != other._shape[i]) {
			return false;
		}
	}

	return true;
}