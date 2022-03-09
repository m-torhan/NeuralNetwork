#include "Utils.h"

uint32_t* genPermutation(uint32_t n) {
	uint32_t i = 0;
	uint32_t j = 0;
	uint32_t tmp;
	uint32_t inversionsCount;
	uint32_t* result;

	result = (uint32_t*)malloc(sizeof(uint32_t) * n);

	for (i = 0; i < n; ++i) {
		result[i] = i;
	}

	srand(time(NULL));

	inversionsCount = 2 * n + rand() % n;

	while (inversionsCount-- != 0) {
		i = rand() % n;
		j = rand() % n;
		tmp = result[i];
		result[i] = result[j];
		result[j] = tmp;
	}

	return result;
}