#ifndef MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_

#include <assert.h>

#include "skel-merge-sort.h"
#include "skel-dac-merge-sort.h"
#include "merge-sort.h"
#include "timer.h"
#include "vector.h"

// Vector factories:
vector<int>   *get_unsorted_int_vector(const size_t size);
vector<float> *get_unsorted_float_vector(const size_t size);

namespace {
// The print_result() function is only called from within templates,
// which are instantiated only when called. As a result, the compiler
// thinks that print_result() is unused. Silence this warning.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

    void print_result(int length, Timer *const t) {
        printf("size: %7u, time: %4ld ms\n", length, t->ms());
    }

#pragma GCC diagnostic pop
}

template<class T>
void test_merge_sort(vector<T> *const in,
                     const unsigned int parallelisation_depth,
                     const typename vector<T>::size_t split_threshold) {

    /*
     * "Standard" 4 phase skeleton procedure.
     */

    // Program definition:
    MergeSort<T> sort(in);

    // Parameters:
    sort.set_parallelisation_depth(parallelisation_depth);
    sort.set_split_threshold(split_threshold);

    // Execution (blocking):
    Timer t;
    sort.run();

    // Results:
    print_result(in->length, &t);
    assert(in->isSorted());

    // Free test data:
    delete[] in->data;
    delete in;
}

template<class T>
void test_sort_func(vector<T> *const in,
                    void (*sort)(T*, T*)) {

    Timer t;
    sort(in->data, in->data + in->length);
    print_result(in->length, &t);

    assert(in->isSorted());

    // Free test data:
    delete[] in->data;
    delete in;
}

template<class T>
void test_sort_func(vector<T> *const in,
                    std::vector<T> (*sort)(std::vector<T>)) {
    std::vector<T> vec(in->data, in->data + in->length);
    vector<T> v;

    Timer t;
    std::vector<T> out = sort(vec);
    print_result(out.size(), &t);

    v.data = &out[0];
    v.length = out.size();

    assert(v.isSorted());
    delete in;
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
