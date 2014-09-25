#ifndef MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_

#include <assert.h>

#include "merge-sort.h"
#include "timer.h"
#include "vector.h"

// Vector factories:
vector<int>   *get_unsorted_int_vector(const size_t size);
vector<float> *get_unsorted_float_vector(const size_t size);

namespace {
    template<class T>
    void print_result(vector<T> *const in, Timer *const t) {
        printf("size: %7u, time: %4ld ms\n", in->length, t->ms());
    }
}

template<class T>
void test_merge_sort(vector<T> *const in,
                     const unsigned int parallelisation_depth) {

    // Setup skeleton:
    MergeSort<T> sort(in);
    sort.set_parallelisation_depth(parallelisation_depth);

    // Timed section:
    Timer t;
    sort.run();
    print_result<T>(in, &t);

    assert(sort.get()->isSorted());
}

template<class T>
void test_sort_func(vector<T> *const in,
                    void (*sort)(T*, T*)) {

    Timer t;
    sort(in->data, in->data + in->length);
    print_result<T>(in, &t);

    assert(in->isSorted());
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
