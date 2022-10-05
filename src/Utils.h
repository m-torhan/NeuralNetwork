#pragma once

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <random>
#include <iostream>
#include <string>
#include <iterator>
#include <sstream>
#include <memory>
#include <stdexcept>

extern double g_time;

/**
 * Defines used for printing time measured between its usage.
 */
#define LOG_TIME {printf("%s - %d : %.6f\n", __FILE__, __LINE__, perf_counter_ns() - g_time); g_time = perf_counter_ns(); }
#define LOG_TIME_NO_PRINT { g_time = perf_counter_ns(); }

/**
 * @brief Produces permutation pattern.
 * 
 * @param n Permutation length.
 * @return Random permutation of given length.
 */
uint32_t* genPermutation(uint32_t n);
/**
 * @brief Samples Normal Distribution.
 * 
 * @return Normal Discribution sample.
 */
float randNormalDistribution();
/**
 * @brief Samples Uniform Distribution.
 * 
 * @param a minimum value.
 * @param b maximum value.
 * @return Uniform Discribution sample.
 */
float randUniform(float a=0, float b=1);
/**
 * @brief Performance timer.
 * 
 * @return System clock in nanoseconds.
 */
double perf_counter_ns();

/**
 * @brief Formats given string.
 * 
 * @tparam Args Type of value that will be inserted in strng.
 * @param format Format string.
 * @param args Values to be insterted in string.
 * @return Formatted string.
 */
template<typename ... Args>
std::string format_string(const std::string& format, Args ... args) {
    int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    if (size_s <= 0){
        throw std::runtime_error("Error during formatting.");
    }
    auto size = static_cast<size_t>(size_s);
    std::unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);

    return std::string(buf.get(), buf.get() + size - 1 );
}

/**
 * @brief Converts vector to string - values in brackets separated by comma.
 * 
 * @tparam T Type of vector entries.
 * @param vector Input vector.
 * @return Vector as a string.
 */
template<typename T>
std::string vector_to_string(std::vector<T> vector) {
    std::ostringstream oss;

    oss << '(';
    if (!vector.empty())
    {
        std::copy(vector.begin(), vector.end()-1,
            std::ostream_iterator<T>(oss, ","));

        oss << vector.back();
    }
    oss << ')';

    return oss.str();
}