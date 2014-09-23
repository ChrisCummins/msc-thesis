#include <algorithm>

#include "list.h"
#include "vector.h"

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

void test_vector_sort(void (*sort)(int*, int*), size_t size) {
    int nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = rand() % size;

    vector<int> *const a = new vector<int>;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    Timer timer;
    sort(a->data, a->data + a->length);
    unsigned int elapsed = timer.ms();

    assert(a->isSorted());

    printf("Time to sort %7u integers: %4u ms\n", a->length, elapsed);
}

void test_vector_sort(void (*sort)(int*, int*),
                      const char *name) {
    int i;
    size_t j;

    std::printf("%s:\n", name);
    for (i = 0, j = 200000; i < 10; i++, j += 200000)
        test_vector_sort(sort, j);
}

int main(int argc, char *argv[]) {
    test_list_sort(std::sort, "std::sort<list>");
    test_list_sort(std::stable_sort, "std::stable_sort<list>");

    test_vector_sort(std::sort, "std::sort<vector>");
    test_vector_sort(std::stable_sort, "std::stable_sort<vector>");

    return 0;
}
