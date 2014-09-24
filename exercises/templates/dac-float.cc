#include "dac.h"

void test_float_sort(const size_t size,
                     const unsigned int parallelisation_depth) {
    float nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = ((float)rand() / 100);

    vector<float> *const a = new vector<float>;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    DC<float> sort(a);
    sort.set_split_degree(2);
    sort.set_parallelisation_depth(parallelisation_depth);

    // Timed section
    Timer t;
    sort.run();
    printf("Time to sort %7lu integers: %4ld ms\n", size, t.ms());

    sort.get()->isSorted();
}

int main(int argc, char *argv[]) {
    unsigned int parallelisation_depth = argc == 2 ? atoi(argv[1]) : 0;

    std::cout << "DC<float>, parallelisation_depth = " << parallelisation_depth << "\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_float_sort(j, parallelisation_depth);

    return 0;
}
