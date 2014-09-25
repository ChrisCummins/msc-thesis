#include "test.h"

int main(int argc, char *argv[]) {
    unsigned int parallelisation_depth = argc == 2 ? atoi(argv[1]) : 0;

    std::cout << "MergeSort<float>, parallelisation_depth = " << parallelisation_depth << "\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_merge_sort<float>(get_unsorted_float_vector(j), parallelisation_depth);

    return 0;
}
