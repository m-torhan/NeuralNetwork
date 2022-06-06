#define EPSILON (0.001f)
#define ASSERT_EQ_EPS(expected, actual) ASSERT_LE(fabs((expected) - (actual)), EPSILON);