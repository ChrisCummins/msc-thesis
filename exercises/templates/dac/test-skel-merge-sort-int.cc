#include "./test.h"  // NOLINT(legal/copyright)

#define TEST_SIZE 1e3

int main(int argc, char *argv[]) {
  test_sort_func<int>(get_unsorted_int_vector(TEST_SIZE), skel::merge_sort);
  return 0;
}
