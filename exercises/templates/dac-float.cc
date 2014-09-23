#include "dac.h"

void test_float_sort(const size_t size,
                     const unsigned int fork_depth) {
    float nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = ((float)rand() / 100);

    vector<float> *const a = new vector<float>;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    Timer t;
    DC<float> sort(a, 2, fork_depth);
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

    std::cout << "DC<float>, fork_depth = " << fork_depth << "\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_float_sort(j, fork_depth);

    return 0;
}
