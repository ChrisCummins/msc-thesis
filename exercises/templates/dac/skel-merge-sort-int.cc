#include "./test.h"  // NOLINT(legal/copyright)

int main(int argc, char *argv[]) {
  std::cout << "skel::merge_sort<int>, "
            << "parallelisation depth = "
            << DAC_SKEL_PARALLELISATION_DEPTH << ", "
            << "split threshold = "
            << SKEL_MERGE_SORT_SPLIT_THRESHOLD << "\n";

  for (int i = 0, j = 50000; i < 40; i++, j += 50000)
    test_sort_func<int>(get_unsorted_int_vector(j), skel::merge_sort);

  return 0;
}
