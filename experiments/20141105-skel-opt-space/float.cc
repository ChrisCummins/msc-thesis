#include "./test.h"  // NOLINT(legal/copyright)

#define TEST_SIZE 1e6

int main(int argc, char *argv[]) {
  test_sort_func<float>(get_unsorted_float_vector(TEST_SIZE), skel::merge_sort);
  return 0;
}
