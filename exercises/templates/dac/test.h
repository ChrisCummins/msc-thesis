#ifndef MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_

#include <assert.h>

#include "skel.h"
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
    print_result<T>(in, &t);
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
    print_result<T>(in, &t);

    assert(in->isSorted());

    // Free test data:
    delete[] in->data;
    delete in;
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_TEST_H_
