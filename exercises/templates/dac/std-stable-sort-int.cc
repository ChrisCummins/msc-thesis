#include <assert.h>
#include <cstdio>
#include <iostream>
#include <algorithm>

#include "vector.h"
#include "timer.h"

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
    test_vector_sort(std::stable_sort, "std::stable_sort<vector>");

    return 0;
}
