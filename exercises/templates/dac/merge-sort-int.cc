#include "test.h"

int main(int argc, char *argv[]) {
    unsigned int parallelisation_depth = argc == 2 ? atoi(argv[1]) : 0;

    std::cout << "MergeSort<int>, parallelisation_depth = " << parallelisation_depth << "\n";
    for (unsigned long i = 0, j = 50000; i < 40; i++, j += 50000)
        test_merge_sort<int>(get_unsorted_int_vector(j), parallelisation_depth);

    return 0;
}
