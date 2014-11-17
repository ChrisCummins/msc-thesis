#include "./test.h"  // NOLINT(legal/copyright)

#define TEST_SIZE 1e3

int main(int argc, char *argv[]) {
  test_sort_func<double>(get_unsorted_double_vector(TEST_SIZE),
                         skel::merge_sort);
  return 0;
}
