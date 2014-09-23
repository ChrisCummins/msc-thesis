#include "dac.h"

void test_int_sort(const size_t size,
                   const unsigned int fork_depth) {
    int nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = rand() % size;

    vector<int> *const a = new vector<int>;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    Timer t;
    DC<int> sort(a, 2, fork_depth);
    printf("Time to sort %7lu integers: %4ld ms\n", size, t.ms());
    sort.get()->isSorted();
}

int main(int argc, char *argv[]) {
    unsigned int fork_depth = 1;

    if (argc == 2) {
        unsigned int i = atoi(argv[1]);
        if (i)
            fork_depth = i;
        else
            std::cout << "warning: <fork_depth> must be an integer. Ignoring argument\n";
    }

    std::cout << "DC<int>, fork_depth = " << fork_depth << "\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_int_sort(j, fork_depth);

    return 0;
}
