#include <algorithm>

#include "list.h"

void test_list_sort(void (*sort)(List::iterator, List::iterator), size_t size) {
    List list = get_unsorted_list(size);

    // Start of timed section
    Timer timer;
    sort(list.begin(), list.end());
    unsigned int elapsed = timer.ms();
    // End of timed section

    assert(list_is_sorted(list));

    printf("Time to sort %7lu integers: %4u ms\n", list.size(), elapsed);
}

void test_list_sort(void (*sort)(List::iterator, List::iterator),
                    const char *name) {
    int i;
    size_t j;

    std::printf("%s:\n", name);
    for (i = 0, j = 200000; i < 10; i++, j += 200000)
        test_list_sort(sort, j);
}

int main(int argc, char *argv[]) {
    test_list_sort(std::stable_sort, "std::stable_sort<list>");

    return 0;
}
