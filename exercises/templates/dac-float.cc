#include "dac.h"

void test_float_sort(const size_t size) {
    float nums[size];

    for (size_t i = 0; i < size; i++)
        nums[i] = ((float)rand() / 100);

    vector<float> *const a = new vector<float>;

    a->data   = &nums[0];
    a->length = sizeof(nums) / sizeof(nums[0]);

    Timer t;
    DC<float> sort(a);
    printf("Time to sort %7lu integers: %4ld ms\n", size, t.ms());
    sort.get()->isSorted();
}

int main(int argc, char *argv[]) {

    std::cout << "DC<float>\n";
    for (unsigned long i = 0, j = 200000; i < 10; i++, j += 200000)
        test_float_sort(j);

    return 0;
}
