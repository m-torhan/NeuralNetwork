#include "Tensor.h"

Tensor::Tensor() {
	// scalar
	_size = 1;
	_shape.push_back(1);
	_data.push_back(0.0f);
}

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

const Tensor& Tensor::operator=(const Tensor& other) {
	_size = other._size;
	_shape = other._shape;
	_data = other._data;

	return *this;
}

const Tensor& Tensor::operator=(const Tensor&& other) {
	_size = other._size;
	_shape = other._shape;
	_data = std::move(other._data);

	return *this;
}

Tensor::~Tensor() {
}

Tensor Tensor::RandomNormal(const std::vector<uint32_t>& shape) {
	Tensor ret(shape);

	for (uint32_t i{ 0 }; i < ret._size; ++i) {
		ret._data[i] = randNormalDistribution();
	}

	return ret;
}

const Tensor Tensor::operator[](std::vector<std::vector<uint32_t> > ranges) const {
	if (ranges.size() > this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Ranges exceed tensor shape. Ranges dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, ranges.size(), this->_shape.size()));
	}
	while (ranges.size() < this->_shape.size()) {
		ranges.push_back(std::vector<uint32_t>{});
	}
	for (uint32_t i{ 0 }; i < ranges.size(); ++i) {
		if (ranges[i].size() > 2) {
			throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. There should be max 2 numbers but are %d.",
				__FILE__, __LINE__, ranges[i].size()));
		}
		if (2 == ranges[i].size()) {
			if (ranges[i][0] >= ranges[i][1]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. First number greater or equal to second. %d >= %d.", i,
					__FILE__, __LINE__, ranges[i][0], ranges[i][1]));
			}
			if (ranges[i][0] >= this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][0]=%d, tensor shape[]=%d.", i, i,
					__FILE__, __LINE__, ranges[i][0], this->_shape[i]));
			}
			if (ranges[i][1] > this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][1]=%d, tensor shape[]=%d.", i, i,
					__FILE__, __LINE__, ranges[i][1], this->_shape[i]));
			}
		}
		if (1 == ranges[i].size()) {
			if (ranges[i][0] >= this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][0]=%d, tensor shape[]=%d.", i, i,
					__FILE__, __LINE__, ranges[i][0], this->_shape[i]));
			}
		}
	}

	std::vector<uint32_t> result_shape;
	std::vector<uint32_t> index;

	for (uint32_t i{ 0 }; i < ranges.size(); ++i) {
		switch (ranges[i].size()) {
			case 0:
				result_shape.push_back(this->_shape[i]);
				index.push_back(0);
				break;
			case 1:
				index.push_back(ranges[i][0]);
				break;
			case 2:
				result_shape.push_back(ranges[i][1] - ranges[i][0]);
				index.push_back(ranges[i][0]);
				break;
		}
	}
	
	if (result_shape.size() == 0) {
		Tensor result;
		result[{ 0 }] = const_cast<const Tensor&>(*this)[index];

		return result;
	}

	Tensor result(result_shape);

	while (true) {
		// set value
		std::vector<uint32_t> sub_index;

		for (uint32_t i{ 0 }; i < index.size(); ++i) {
			if (2 == ranges[i].size()) {
				sub_index.push_back(index[i] - ranges[i][0]);
			}
			else if (0 == ranges[i].size()) {
				sub_index.push_back(index[i]);
			}
		}
		
		result[sub_index] = (*this)[index];

		// increment index
		bool inc_next{ false };
		for (int32_t i{ static_cast<int32_t>(index.size() - 1) }; i >= 0; --i) {
			if (1 != ranges[i].size()) {
				++index[i];
			}
			if (0 == ranges[i].size()) {
				if (index[i] >= this->_shape[i]) {
					// overflow
					index[i] = 0;
					inc_next = true;
				}
				else {
					inc_next = false;
					break;
				}
			}
			else if (2 == ranges[i].size()) {
				if (index[i] >= ranges[i][1]) {
					// overflow
					index[i] = ranges[i][0];
					inc_next = true;
				}
				else {
					inc_next = false;
					break;
				}
			}
		}

		if (inc_next) {
			break;
		}
	}

	return result;
}

const float Tensor::operator[](const std::vector<uint32_t>& index) const {
	uint32_t flat_index{ 0 };
	uint32_t sub_size{ this->_size };

	if (index.size() != this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Indices dim is different than tensor dim. Indices dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, index.size(), this->_shape.size()));
	}

	for (uint32_t i{ 0 }; i < this->_shape.size(); ++i) {
		sub_size /= this->_shape[i];
		uint32_t sub_index{ index[i] };
		if (sub_index >= this->_shape[i]) {
			throw std::invalid_argument(format_string("%s %d : Index at %d exceeds tensor dimension. INdex[%d]=%d, tensor shape[]=%d.",
				__FILE__, __LINE__, i, i, index[i], this->_shape[i]));
		}
		flat_index += sub_size * sub_index;
	}

	return this->_data[flat_index];
}

TensorSlice Tensor::operator[](std::vector<std::vector<uint32_t> > ranges) {
	if (ranges.size() > this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Ranges exceed tensor shape. Ranges dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, ranges.size(), this->_shape.size()));
	}
	while (ranges.size() < this->_shape.size()) {
		ranges.push_back(std::vector<uint32_t>{});
	}
	for (uint32_t i{ 0 }; i < ranges.size(); ++i) {
		if (ranges[i].size() > 2) {
			throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. There should be max 2 numbers but are %d.",
				__FILE__, __LINE__, ranges[i].size()));
		}
		if (2 == ranges[i].size()) {
			if (ranges[i][0] >= ranges[i][1]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. First number greater or equal to second. %d >= %d.",
					__FILE__, __LINE__, i, ranges[i][0], ranges[i][1]));
			}
			if (ranges[i][0] >= this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][0]=%d, tensor shape[]=%d.",
					__FILE__, __LINE__, i, i, ranges[i][0], this->_shape[i]));
			}
			if (ranges[i][1] > this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][1]=%d, tensor shape[]=%d.",
					__FILE__, __LINE__, i, i, ranges[i][1], this->_shape[i]));
			}
		}
		if (1 == ranges[i].size()) {
			if (ranges[i][0] >= this->_shape[i]) {
				throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][0]=%d, tensor shape[]=%d.",
					__FILE__, __LINE__, i, i, ranges[i][0], this->_shape[i]));
			}
		}
	}

	return TensorSlice(*this, ranges);
}

TensorCell Tensor::operator[](const std::vector<uint32_t>& index) {
	if (index.size() != this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Indices dim is different than tensor dim. Indices dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, index.size(), this->_shape.size()));
	}

	for (uint32_t i{ 0 }; i < this->_shape.size(); ++i) {
		if (index[i]  >= this->_shape[i]) {
			throw std::invalid_argument(format_string("%s %d : Index at %d exceeds tensor dimension. INdex[%d]=%d, tensor shape[]=%d.",
				__FILE__, __LINE__, i, i, index[i], this->_shape[i]));
		}
	}

	return TensorCell(*this, index);
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

void Tensor::setValues(const std::vector<float>& values) {
	if (this->_data.size() != values.size()) {
		throw std::invalid_argument(format_string("%s %d : Provided values vector has wrong size. Values size=%d, tensor size=%d.",
			__FILE__, __LINE__, values.size(), this->_data.size()));
	}
	this->_data = values;
}

const Tensor Tensor::operator-() const {
	Tensor result{ *this };
	return result*(-1.0f);
}

Tensor& Tensor::operator+=(const Tensor& other) {
	if (((this->_shape.size() < other._shape.size()) ||
		 (!this->validateShapeReversed(other))) &&
		(1 != other._size)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	#ifndef SSE
	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		this->_data[i] += other._data[i % other._size];
	}
	#else	// SSE
	if (this->_size == other._size) {
		SSE_vector_add(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (1 == other._size) {
		SSE_tensor_add_scalar(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (this->validateShapeReversed(other)) {
		SSE_tensor_add(this->_size, this->_data.data(), other._size, other._data.data(), this->_data.data());
	} else {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator+(const Tensor& other) const {
	Tensor result{ *this };
	result += other;
	return result;
}

Tensor& Tensor::operator+=(float number) {
	#ifndef SSE
	for (uint32_t i{ 0 } ; i < this->_size; ++i) {
		this->_data[i] += number;
	}
	#else	// SSE
	SSE_tensor_add_scalar(this->_size, this->_data.data(), &number, this->_data.data());
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator+(float number) const {
	Tensor result{ *this };
	result += number;
	return result;
}

const Tensor operator+(float number, const Tensor& other) {
	Tensor result{ other };
	result += number;
	return result;
}

Tensor& Tensor::operator-=(const Tensor& other) {
	if (((this->_shape.size() < other._shape.size()) ||
		 (!this->validateShape(other))) &&
		(1 != other._size)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	#ifndef SSE
	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		this->_data[i] -= other._data[i % other._size];
	}
	#else	// SSE
	if (this->_size == other._size) {
		SSE_vector_sub(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (1 == other._size) {
		SSE_tensor_sub_scalar(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (this->validateShapeReversed(other)) {
		SSE_tensor_sub(this->_size, this->_data.data(), other._size, other._data.data(), this->_data.data());
	} else {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator-(const Tensor& other) const {
	Tensor result{ *this };
	result -= other;
	return result;
}

Tensor& Tensor::operator-=(float number) {
	#ifndef SSE
	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		this->_data[i] -= number;
	}
	#else	// SSE
	SSE_tensor_sub_scalar(this->_size, this->_data.data(), &number, this->_data.data());
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator-(float number) const {
	Tensor result{ *this };
	result -= number;
	return result;
}

const Tensor operator-(float number, const Tensor& other) {
	Tensor result{ -other };
	result += number;
	return result;
}

Tensor& Tensor::operator*=(const Tensor& other) {
	if (((this->_shape.size() < other._shape.size()) ||
		 (!this->validateShapeReversed(other))) &&
		(1 != other._size)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	#ifndef SSE
	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		this->_data[i] *= other._data[i % other._size];
	}
	#else	// SSE
	if (this->_size == other._size) {
		SSE_vector_mul(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (1 == other._size) {
		SSE_tensor_mul_scalar(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (this->validateShapeReversed(other)) {
		SSE_tensor_mul(this->_size, this->_data.data(), other._size, other._data.data(), this->_data.data());
	} else {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator*(const Tensor& other) const {
	Tensor result{ *this };
	result *= other;
	return result;
}

Tensor& Tensor::operator*=(float number) {
	#ifndef SSE
	for (uint32_t i{ 0 } ; i < this->_size; ++i) {
		this->_data[i] *= number;
	}
	#else	// SSE
	std::vector<float> tmp(this->_data.size());
	SSE_tensor_mul_scalar(this->_size, this->_data.data(), &number, tmp.data());
	this->_data = tmp;
	#endif	// SSE
	return *this;
}

const Tensor Tensor::operator*(float number) const {
	Tensor result{ *this };
	result *= number;
	return result;
}

const Tensor operator*(float number, const Tensor& other) {
	Tensor result{ other };
	result *= number;
	return result;
}

Tensor& Tensor::operator/=(const Tensor& other) {
	if (((this->_shape.size() < other._shape.size()) ||
		 (!this->validateShape(other))) &&
		(1 != other._size)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	#ifndef SSE
	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		this->_data[i] /= other._data[i % other._size];
	}
	#else	// SSE
	if (this->_size == other._size) {
		SSE_vector_div(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (1 == other._size) {
		SSE_tensor_div_scalar(this->_size, this->_data.data(), other._data.data(), this->_data.data());
	}
	else if (this->validateShapeReversed(other)) {
		SSE_tensor_div(this->_size, this->_data.data(), other._size, other._data.data(), this->_data.data());
	} else {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator/(const Tensor& other) const {
	Tensor result{ *this };
	result /= other;
	return result;
}

Tensor& Tensor::operator/=(float number) {
	#ifndef SSE
	for (uint32_t i{ 0 } ; i < this->_size; ++i) {
		this->_data[i] /= number;
	}
	#else	// SSE
	SSE_tensor_div_scalar(this->_size, this->_data.data(), &number, this->_data.data());
	#endif	// SSE

	return *this;
}

const Tensor Tensor::operator/(float number) const {
	Tensor result = *this;
	result /= number;
	return result;
}

const Tensor operator/(float number, const Tensor& other) {
	Tensor result = other;

	#ifndef SSE
	for (uint32_t i{ 0 } ; i < other._size; ++i) {
		result._data[i] = number / result._data[i];
	}
	#else	// SSE
	SSE_scalar_div_tensor(&number, other._size, other._data.data(), result._data.data());
	#endif	// SSE

	return result;
}

const Tensor Tensor::operator==(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] == other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator!=(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] != other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator>(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] > other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator>=(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] >= other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator<(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] < other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator<=(const Tensor& other) const {
	if ((this->_shape.size() < other._shape.size()) ||
		!this->validateShape(other)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}

	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] <= other._data[i % other._size] ? 1.0f: 0.0f;
	}

	return result;
}

const Tensor Tensor::operator==(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] == number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::operator!=(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] != number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::operator>(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] > number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::operator>=(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] >= number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::operator<(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] < number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::operator<=(float number) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = this->_data[i] <= number ? 1.0f : 0.0f;
	}

	return result;
}

const Tensor Tensor::addPadding(const std::vector<uint32_t>& axes, const std::vector<Padding>& paddings, const std::vector<uint32_t>& counts) const {
	if (axes.size() != paddings.size() || axes.size() != counts.size()) {
		throw std::invalid_argument(format_string("%s %d : Provided arguments have wrong dim. Axes dim=%d, paddings dim=%d, counts dim=%d. All should be equal.",
			__FILE__, __LINE__, axes.size(), paddings.size(), counts.size()));
	}

	if (axes.size() > this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Provided axes have wrong dim. Axes dim=%d, Tensor dim=%d.",
			__FILE__, __LINE__, axes.size(), this->_shape.size()));
	}

	std::vector<uint32_t> result_shape{ this->_shape };

	for (uint32_t i{ 0 }; i < axes.size(); ++i) {
		result_shape[axes[i]] += counts[i]*(!!(paddings[i] & Left) + !!(paddings[i] & Right));
	}

	Tensor result(result_shape);

	result *= 0.0f;

	std::vector<std::vector<uint32_t> > ranges(this->_shape.size());

	for (uint32_t i{ 0 }; i < axes.size(); ++i) {
		ranges[axes[i]].push_back(counts[i]*(!!(paddings[i] & Left)));
		ranges[axes[i]].push_back(this->_shape[axes[i]] + counts[i]*(!!(paddings[i] & Left)));
	}

	result[ranges] = *this;

	return result;
}

const Tensor Tensor::dotProduct(const Tensor& other) const {
	if (this->_shape.size() == 1 && other._shape.size() == 1) {
		// vector inner product
		if (this->_shape[0] != other._shape[0]) {
			throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
				__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
		}
		Tensor result;

		#ifndef SSE
		result *= 0.0f;
		
		for (uint32_t i{ 0 }; i < this->_size; ++i) {
			result._data[0] += this->_data[i] * other._data[i];
		}
		#else
		float result_value{ 0.0f };

		SSE_vector_inner_product(this->_size, this->_data.data(), other._data.data(), &result_value);

		result._data[0] = result_value;
		#endif

		return result;
	}

	if (this->_shape.size() == 2 && other._shape.size() == 2) {
		// matrix multiplication
		if (this->_shape[1] != other._shape[0]) {
			throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
				__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
		}
		std::vector<uint32_t> result_shape = { this->_shape[0], other._shape[1] };

		Tensor result(result_shape);

		for (uint32_t i{ 0 }; i < result_shape[0]; ++i) {
			for (uint32_t j{ 0 }; j < result_shape[1]; ++j) {
				for (uint32_t k{ 0 }; k < this->_shape[1]; ++k) {
					result._data[i * result_shape[1] + j] += this->_data[i * this->_shape[1] + k] * other._data[k * other._shape[1] + j];
				}
			}
		}

		return result;
	}

	if (this->_shape.size() == 0) {
		// scalar multiplication
		return other*this->_data[0];
	}

	if (other._shape.size() == 0) {
		// scalar multiplication
		return (*this)*other._data[0];
	}

	if (other._shape.size() == 1) {
		// sum product over the last axis of this and other (vector)
		if (this->_shape[this->_shape.size() - 1] != other._shape[0]) {
			throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
				__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
		}
		std::vector<uint32_t> result_shape{ this->_shape };
		result_shape.pop_back();

		Tensor result(result_shape);

		uint32_t d_i{ this->_size / other._shape[0] };

		for (uint32_t i{ 0 }; i < other._shape[0]; ++i) {
			for (uint32_t j{ 0 }; j < d_i; ++j) {
				result._data[j] += this->_data[i * d_i + j] * other._data[i];
			}
		}

		return result;
	}

	else {
		// sum product over the last axis of this and the second-to-last axis of other
		if (this->_shape[this->_shape.size()  - 1] != other._shape[other._shape.size() - 2]) {
			throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
				__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
		}

		Tensor result;

		//d_i = this->_size / this->_shape[this->_shape.size()  - 1];
		//d_j = other._size / other._shape[other._shape.size()  - 2];

		// TODO (dot product for higher dimensions currently not essential)

		return result;
	}
}

const Tensor Tensor::dotProductTranspose(const Tensor& other) const {
	if ((this->_shape.size() != 2) || (other._shape.size() != 2)) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	// matrix multiplication
	if (this->_shape[1] != other._shape[1]) {
		throw std::invalid_argument(format_string("%s %d : Provided operands has wrong shapes. First operand shape=%s, second operand shape=%s",
			__FILE__, __LINE__, vector_to_string(this->_shape).c_str(), vector_to_string(other._shape).c_str()));
	}
	std::vector<uint32_t> result_shape = { this->_shape[0], other._shape[0] };

	Tensor result(result_shape);

	#ifndef SSE
	for (uint32_t i{ 0 }; i < result_shape[0]; ++i) {
		for (uint32_t j{ 0 }; j < result_shape[1]; ++j) {
			for (uint32_t k{ 0 }; k < this->_shape[1]; ++k) {
				result._data[i * result_shape[1] + j] += this->_data[i * this->_shape[1] + k] * other._data[j * other._shape[1] + k];
			}
		}
	}
	#else	// SSE
	for (uint32_t i{ 0 }; i < result_shape[0]; ++i) {
		SSE_tensor_dot_product_transpose(1, result_shape[1], this->_shape[1], this->_data.data(), other._data.data(), result._data.data() + i*result_shape[1]);
	}
	#endif	// SSE

	return result;
}

const Tensor Tensor::tensorProduct(const Tensor& other) const {
	std::vector<uint32_t> result_shape{ this->_shape };
	result_shape.insert(result_shape.end(), other._shape.begin(), other._shape.end());

	Tensor result(result_shape);

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		Tensor subresult = other;
		subresult *= this->_data[i];
		std::copy(subresult._data.begin(), subresult._data.end(), result._data.begin() + i * subresult._data.size());
	}

	return result;
}

const Tensor Tensor::applyFunction(float (*function)(float)) const {
	Tensor result{ *this };

	for (uint32_t i{ 0 }; i < this->_size; ++i) {
		result._data[i] = function(result._data[i]);
	}

	return result;
}

const Tensor Tensor::flatten(uint32_t start_axis) const {
	if (start_axis >= this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Provided axis exceeds tensor dim. axis=%d, Tensor dim=%d",
			__FILE__, __LINE__, start_axis, this->_shape.size()));
	}

	uint32_t subsize{ 1 };
	for (uint32_t i{ start_axis }; i < this->_shape.size() ; ++i) {
		subsize *= this->_shape[i];
	}

	std::vector<uint32_t> result_shape;

	for (uint32_t i{ 0 }; i < start_axis; ++i) {
		result_shape.push_back(this->_shape[i]);
	}
	result_shape.push_back(subsize);

	Tensor result(result_shape);

	std::copy(this->_data.begin(), this->_data.end(), result._data.begin());

	return result;
}

const Tensor Tensor::sum(uint32_t axis) const {
	if (axis >= this->_shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Provided axis exceeds tensor dim. Axis=%d, Tensor dim=%d",
			__FILE__, __LINE__, axis, this->_shape.size()));
	}

	std::vector<uint32_t> result_shape;

	for (uint32_t i{ 0 }; i < _shape.size() ; ++i) {
		if (i != axis) {
			result_shape.push_back(_shape[i]);
		}
	}

	Tensor result(result_shape);

	uint32_t d_i{ 1 };
	for (uint32_t i{ axis + 1 }; i < this->_shape.size() ; ++i) {
		d_i *= this->_shape[i];
	}
	#ifndef SSE

	uint32_t d_j{ axis == this->_shape.size() - 1 ? this->_shape[this->_shape.size()  - 1] : 1 };

	uint32_t d_k{ d_i * this->_shape[axis] };

	for (uint32_t k{ 0 }; k < this->_size / d_k; ++k) {
		for (uint32_t j{ 0 }; j < d_i; ++j) {
			for (uint32_t i{ 0 }; i < this->_shape[axis] ; ++i) {
				result._data[j + d_i * k] += this->_data[d_i * i + d_j * j + d_k * k];
			}
		}
	}
	#else 	// SSE
	if (axis == this->_shape.size() - 1) {
		SSE_tensor_last_axis_sum(this->_size / (d_i * this->_shape[axis]), this->_shape[axis], this->_data.data(), result._data.data());
	}
	else {
		SSE_tensor_axis_sum(this->_size / (d_i * this->_shape[axis]), d_i, this->_shape[axis], this->_data.data(), result._data.data());
	}
	#endif	// SSE

	return result;
}

float Tensor::sum() const {
	float result{ 0.0f };
	
	#ifndef SSE
	for (auto value : _data) {
		result += value;
	}
	#else 	// SSE
	SSE_tensor_sum(this->_size, this->_data.data(), &result);
	#endif	// SSE

	return result;
}

float Tensor::max() const {
	float result = _data[0];

	for (auto value : _data) {
		if (value > result) {
			result = value;
		}
	}

	return result;
}

float Tensor::mean() const {
	float sum = this->sum();

	return sum/_size;

}

const Tensor Tensor::transpose() const {
	if (this->_shape.size() != 2) {
		throw std::invalid_argument(format_string("%s %d : Only transpose of tensors with dim equal 2 is supported. Tensor dim=%d",
			__FILE__, __LINE__, this->_shape.size()));
	}

	std::vector<uint32_t> result_shape = { this->_shape[1], this->_shape[0] };

	Tensor result(result_shape);

	for (uint32_t i{ 0 }; i < result_shape[0]; ++i) {
		for (uint32_t j{ 0 }; j < result_shape[1]; ++j) {
			result._data[i * result_shape[1] + j] = this->_data[j * result_shape[0] + i];
		}
	}

	return result;
}

const Tensor Tensor::shuffle() const {
	uint32_t axis{ 0 }; // currently only for first axis

	Tensor result{ *this };
	uint32_t subsize{ this->_size/this->_shape[axis] };
	float *tmp{ new float[subsize] };

	srand(time(NULL));

	uint32_t shuffle_count{ (this->_shape[axis] + rand() % this->_shape[axis]) << 1 };

	for (uint32_t i{ 0 }; i < shuffle_count; ++i) {
		uint32_t rand_a = rand() % this->_shape[axis];
		uint32_t rand_b = rand() % this->_shape[axis];
		if (rand_a == rand_b) {
			continue;
		}

		std::memcpy(tmp, &result._data[subsize * rand_a], sizeof(float) * subsize);
		std::memcpy(&result._data[subsize * rand_a], &result._data[subsize * rand_b], sizeof(float) * subsize);
		std::memcpy(&result._data[subsize * rand_b], tmp, sizeof(float) * subsize);
	}

	delete[] tmp;

	return result;
}

const Tensor Tensor::shuffle(uint32_t *pattern) const {
	uint32_t axis{ 0 }; // currently only for first axis

	Tensor result{ *this };

	uint32_t subsize{ this->_size / this->_shape[axis] };

	float *tmp{ new float[subsize] };
	uint32_t *shuffled{ new uint32_t[this->_shape[axis]]};

	std::fill(shuffled, shuffled + this->_shape[axis], 0);

	for (uint32_t i{ 0 }; i < this->_shape[axis]; ++i) {
		if (!shuffled[i]) {
			shuffled[i] = 1;
			uint32_t j{ pattern[i] };
			do {
				shuffled[j] = 1;
				
				std::memcpy(tmp, &result._data[subsize * j], sizeof(float) * subsize);
				std::memcpy(&result._data[subsize * j], &result._data[subsize * pattern[j]], sizeof(float) * subsize);
				std::memcpy(&result._data[subsize * pattern[j]], tmp, sizeof(float) * subsize);
			
				j = pattern[j];
			} while (j != i);
		}
	}

	delete[] tmp;
	delete[] shuffled;

	return result;
}

const Tensor Tensor::reshape(std::vector<uint32_t> new_shape) const {
	uint32_t new_size{ 1 };
	for (auto s : new_shape) {
		new_size *= s;
	}
	if (new_size != this->_size) {
		throw std::invalid_argument(format_string("%s %d : Provided new shape changes size of tensor. New size=%d, old size=%d.",
			__FILE__, __LINE__, new_size, this->_size));
	}

	Tensor result{ *this };

	result._shape = std::move(new_shape);

	return result;
}

void Tensor::print() const {
	printf("Tensor({ ");
	for (uint32_t i{ 0 }; i < this->_shape.size(); ++i) {
		printf("%d ", this->_shape[i]);
	}
	printf("})\n");
}

bool Tensor::validateShape(const Tensor& other) const {
	uint32_t min_dim{ this->_shape.size() < other._shape.size() ? this->_shape.size() : other._shape.size() };
	
	for (uint32_t i{ 0 }; i < min_dim; ++i) {
		if (this->_shape[i] != other._shape[i]) {
			return false;
		}
	}

	return true;
}

bool Tensor::validateShapeReversed(const Tensor& other) const {
	uint32_t min_dim{ this->_shape.size() < other._shape.size() ? this->_shape.size() : other._shape.size() };
	
	for (uint32_t i{ 0 }; i < min_dim; ++i) {
		if (this->_shape[this->_shape.size() - i - 1] != other._shape[other._shape.size() - 1 - i]) {
			return false;
		}
	}

	return true;
}

const Tensor& TensorSlice::operator=(const Tensor& other) {
	if (_slice_ranges.size() != _tensor._shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Ranges exceed tensor shape. Ranges dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, _slice_ranges.size(), _tensor._shape.size()));
	}
	for (uint32_t i{ 0 }; i < _slice_ranges.size(); ++i) {
		switch (_slice_ranges[i].size()) {
			case 0:
				break;
			case 1:
				if (_slice_ranges[i][0] >= _tensor._shape[i]) {
				}
				break;
			case 2:
				if (_slice_ranges[i][0] >= _slice_ranges[i][1]) {
					throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. First number greater or equal to second. %d >= %d.",
						__FILE__, __LINE__, i, _slice_ranges[i][0], _slice_ranges[i][1]));
				}
				if (_slice_ranges[i][0] > _tensor._shape[i]) {
					throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][0]=%d, tensor shape[]=%d.",
						__FILE__, __LINE__, i, i, _slice_ranges[i][0], _tensor._shape[i]));
				}
				if (_slice_ranges[i][1] > _tensor._shape[i]) {
					throw std::invalid_argument(format_string("%s %d : Range at %d exceeds tensor dimension. Ranges[%d][1]=%d, tensor shape[]=%d.",
						__FILE__, __LINE__, i, i, _slice_ranges[i][1], _tensor._shape[i]));
				}
				break;
			default:
				throw std::invalid_argument(format_string("%s %d : Range at %d has wrong format. There should be max 2 numbers but are %d.",
					__FILE__, __LINE__, _slice_ranges[i].size()));
				break;
		}
	}

	std::vector<uint32_t> index(_tensor._shape.size());

	for (uint32_t i{ 0 }; i < index.size(); ++i) {
		if (0 == _slice_ranges[i].size()) {
			index[i] = 0;
		}
		else {
			index[i] = _slice_ranges[i][0];
		}
	}

	while (true) {
		// set value
		std::vector<uint32_t> sub_index;

		for (uint32_t i{ 0 }; i < index.size(); ++i) {
			if (2 == _slice_ranges[i].size()) {
				sub_index.push_back(index[i] - _slice_ranges[i][0]);
			}
			else if (0 == _slice_ranges[i].size()) {
				sub_index.push_back(index[i]);
			}
		}
		
		_tensor[index] = other[sub_index];

		// increment index
		bool inc_next = false;
		for (int32_t i{ static_cast<int32_t>(index.size() - 1) }; i >= 0; --i) {
			if (0 == _slice_ranges[i].size()) {
				++index[i];
				if (index[i] >= _tensor._shape[i]) {
					// overflow
					index[i] = 0;
					inc_next = true;
				}
				else {
					inc_next = false;
					break;
				}
			}
			else if (2 == _slice_ranges[i].size()) {
				++index[i];
				if (index[i] >= _slice_ranges[i][1]) {
					// overfloww
					index[i] = _slice_ranges[i][0];
					inc_next = true;
				}
				else {
					inc_next = false;
					break;
				}
			}
		}

		if (inc_next) {
			break;
		}
	}

	return other;
}

float TensorCell::operator=(float value) {
	if (_cell_index.size() != _tensor._shape.size()) {
		throw std::invalid_argument(format_string("%s %d : Indices dim is different than tensor dim. Indices dim=%d, tensor dim=%d.",
			__FILE__, __LINE__, _cell_index.size(), _tensor._shape.size()));
	}

	uint32_t flat_idx{ 0 };
	uint32_t subsize = _tensor._size;
	for (uint32_t i{ 0 }; i < _tensor._shape.size(); ++i) {
		subsize /= _tensor._shape[i];
		uint32_t sub_index{ _cell_index[i] };
		if (sub_index >= _tensor._shape[i]) {
			throw std::invalid_argument(format_string("%s %d : Index at %d exceeds tensor dimension. INdex[%d]=%d, tensor shape[]=%d.",
				__FILE__, __LINE__, i, i, _cell_index[i], _tensor._shape[i]));
		}
		flat_idx += subsize * _cell_index[i];
	}

	_tensor._data[flat_idx] = value;

	return value;
}