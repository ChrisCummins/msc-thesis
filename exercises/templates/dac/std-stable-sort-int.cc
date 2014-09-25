#include <algorithm>

#include "test.h"

int main(int argc, char *argv[]) {
    int i;
    size_t j;

    std::cout << "std::stable_sort<int>\n";
    for (i = 0, j = 200000; i < 10; i++, j += 200000)
        test_sort_func(get_unsorted_int_vector(j), std::stable_sort);

    return 0;
}
