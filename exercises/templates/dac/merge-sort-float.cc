#include "./test.h"

int main(int argc, char *argv[]) {
  const unsigned int parallelisation_depth = argc >= 2 ? atoi(argv[1]) : 0;
  const vector<int>::size_t split_threshold = argc >= 3 ? atoi(argv[2]) : 100;

  std::cout << "MergeSort<float>, "
            << "parallelisation_depth = " << parallelisation_depth << ", "
            << "split_threshold = " << split_threshold << "\n";

  for (int i = 0, j = 50000; i < 40; i++, j += 50000)
    test_merge_sort<float>(get_unsorted_float_vector(j), parallelisation_depth,
                           split_threshold);

  return 0;
}
