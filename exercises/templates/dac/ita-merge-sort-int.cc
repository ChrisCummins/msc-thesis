#include "ita-merge-sort.h"

#include "test.h"

int main(int argc, char *argv[]) {

    std::cout << "ita_merge_sort<int>\n";
    for (unsigned long i = 0, j = 50000; i < 40; i++, j += 50000)
        test_sort_func(get_unsorted_int_vector(j), ita_merge_sort<int>);

    return 0;
}
