#include <gtest/gtest.h>
#include "src/Utils.h"

TEST(Utils_test, GenPermutationResultShouldContainAllNumbersUpToN) {
    bool b[32] = { false };
    uint32_t* permutation;
    int s = 0;
    int i = 0;

    permutation = genPermutation(32);

    for (i = 0; i < 32; ++i) {
        b[permutation[i]] = true;
    }
    for (i = 0; i < 32; ++i) {
        s += b[i];
    }
    ASSERT_EQ(32, s);
}