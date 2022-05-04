#pragma once

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <ctime>
#include <random>

extern double g_time;

#define LOG_TIME {printf("%s - %d : %.6f\n", __FILE__, __LINE__, perf_counter_ns() - g_time); g_time = perf_counter_ns(); }

uint32_t* genPermutation(uint32_t n);
float randNormalDistribution();
float randUniform(float a=0, float b=1);
double perf_counter_ns();