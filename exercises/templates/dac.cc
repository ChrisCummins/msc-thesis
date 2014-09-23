#include "dac.h"

typedef DC<int>      MergeSort;
typedef vector<int>  vector_t;

void test_sort(const size_t size) {
    int nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = rand() % size;

    vector_t *const a = new vector_t;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    Timer t;
    MergeSort sort(a);
    printf("Time to sort %7lu integers: %4ld ms\n", size, t.ms());
    sort.get()->isSorted();
}

int main(int argc, char *argv[]) {

    std::cout << "DC<int>\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_sort(j);

    return 0;
}
