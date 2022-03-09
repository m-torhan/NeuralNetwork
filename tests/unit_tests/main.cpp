#include <gtest/gtest.h>
#ifdef USE_MOCKS
    #include <gmock/gmock.h>
#endif

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
#ifdef USE_MOCKS
    ::testing::InitGoogleMock(&argc, argv);
#endif

    return RUN_ALL_TESTS();
}