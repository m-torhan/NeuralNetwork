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

#define LOG_TIME {printf("%s - %d : %.6f\n", __FILE__, __LINE__, perf_counter_ns() - g_time); g_time = perf_counter_ns(); }
#define LOG_TIME_NO_PRINT { g_time = perf_counter_ns(); }

uint32_t* genPermutation(uint32_t n);
float randNormalDistribution();
float randUniform(float a=0, float b=1);
double perf_counter_ns();

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