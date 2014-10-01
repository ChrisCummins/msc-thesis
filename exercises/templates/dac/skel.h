#ifndef MSC_THESIS_EXERCISES_TEMPLATES_SKEL_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_SKEL_H_

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

#include "debug.h"

namespace skel {

/*
 * Fixed Degree Divide and Conquer skeleton.
 *
 * Provides a template for writing divide and conquer solutions by
 * defining the 4 "muscle" functions:
 *
 * bool is_indivisible(T *in)
 *
 *    Returns whether the problem "in" can be split into multiple
 *    sub-problems, or whether the problem should be conquered
 *    directly.
 *
 * T *divide(T *const in);
 *
 *    Returns a pointer to an array of sub-problems that have are the
 *    result of dividing "in". This function is responsible for
 *    allocating the required memory on the heap.
 *
 * void conquer(T *const in)
 *
 *    Solve an indivisible problem "in" directly, and in-place.
 *
 * void combine(T *const in, T *const out)
 * 
 *    Combines the array of sub-problems pointed to by "in" into the
 *    "out" parameter. The size of input array "in" is determined by
 *    the fixed degree value.
 *
 * divide_and_conquer() acts as the "worker" function, invoking the
 * muscle functions as required. When invoked, it determines whether
 * the size of the input is small enough to "conquer" directly, and if
 * not, it uses the "divide" muscle function to split the problem into
 * "degree" sub-problems.
 *
 * Primitive concurrency is provided by measuring the depth of
 * recursion in the divide_and_conquer() function. If the depth is
 * less than the "parallelisation_depth" parameter, then recursion
 * will occur in a new thread. This means that the maximum total
 * number of threads created is coarsely determined by both the degree
 * and the parallelisation depth. For a fixed degree of 2:
 *
 *             parallelisation_depth   max_no_of_threads
 *                                 0   1
 *                                 1   3
 *                                 2   7
 *                                 3   15
 *                                 4   31
 *
 * Or, more generally:
 *
 *     n^k + (n-1)^k + (n-2)^k + ... + 1
 *
 * Where "n" is the parallelisation depth, and "k" is the degree.
 */

template<typename T,
        const    uint degree,
        const    uint parallelisation_depth,
        bool     is_indivisible(T *const in),
        T       *divide(T *const in),
        void     conquer(T *const in),
        void     combine(T *const in, T *const out)>
void divide_and_conquer(T *const in, const uint depth);


/*
 * A concrete stable merge sort implementation, using the Divide and
 * Conquer skeleton.
 *
 * The only muscle function we need to provide is merge(), since the
 * defaults for the FDDC template are satisfactory.
 *
 * MergeSort requires that the class "T" supports comparion and
 * equality operators.
 */
template<typename T>
void merge_sort(T *const start, T *const end);


#define SKEL_MERGE_SORT_PARALLELISATION_DEPTH 2
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD       100


/*
 * Generic data type for storing pointers to contiguous arrays.
 */
template<typename T>
class list {
 public:
    T *start;
    T *end;

    list() {};
    list(T *const start, T *const end) : start(start), end(end) {};
};


/***********************************************/
/* Divide and Conquer Skeleton implementations */
/***********************************************/

template<typename T,
         const    uint degree,
         const    uint parallelisation_depth,
         bool     is_indivisible(T *const in),
         T       *divide(T *const in),
         void     conquer(T *const in),
         void     combine(T *const in, T *const out)>
void divide_and_conquer(T *const in, const uint depth) {

// Since the template syntax has rendered the parameterised name of
// this function unholy ugly, we'll do a cheeky macro def to keep
// things looking pretty and consistent (don't worry ma, we'll tidy up
// when we're done playing).
#define template_params T, degree, parallelisation_depth, is_indivisible, divide, conquer, combine
#define self divide_and_conquer<template_params>

    // Determine whether we're in a base case or recursion case:
    if (is_indivisible(in)) {

        // If we can solve the problem directly, then do that:
        conquer(in);

    } else {

        const unsigned int next_depth = depth + 1;

        // Split our problem into "k" sub-problems:
        T *const split = divide(in);

        /*
         * Recurse and solve for all sub-problems created by divide().
         *
         * If the depth is less than "parallelisation_depth", then we
         * create a new thread to perform the recursion in. Otherwise,
         * we recurse sequentially.
         */
        if (depth < parallelisation_depth) {
            std::thread threads[degree];

            // Create threads:
            for (uint i = 0; i < degree; i++) {
                threads[i] = std::thread(self, &split[i], next_depth);
                DAC_DEBUG_PRINT(3, "Creating thread at depth " << next_depth);
            }

            // Block until threads complete:
            for (auto &thread : threads) {
                thread.join();
                DAC_DEBUG_PRINT(3, "Thread completed at depth " << next_depth);
            }

        } else {

            // Sequential (*yawn*):
            for (uint i = 0; i < degree; i++)
                self(&split[i], next_depth);

        }

        // Merge the conquered "k" sub-problems into a solution:
        combine(split, in);

        // Free heap memory:
        delete[] split;
    }

#undef template_params
#undef self
}


/***************************************/
/* Merge sort skeleton implementations */
/***************************************/

template<typename T>
bool is_indivisible(list<T> *const in) {
    return (in->end - in->start) <= SKEL_MERGE_SORT_SPLIT_THRESHOLD;
}


template<typename T, const unsigned int n>
list<T> *divide(list<T> *const in) {
    list<T> *const out = new list<T>[n];

    const uint input_length = in->end - in->start;
    const uint subproblem_length = input_length / n;
    const uint first_subproblem_length = input_length - (n - 1) * subproblem_length;

    // Split "in" into "k" vectors, starting at address "out".
    out[0].start = in->start;
    out[0].end = in->start + first_subproblem_length;

    for (uint i = 1; i < n; i++) {
        const uint start = (i-1) * subproblem_length + first_subproblem_length;

        out[i].start = &in->start[start];
        out[i].end = &in->start[start] + subproblem_length;
    }

    return out;
}


template<typename T>
void insertion_sort(list<T> *const in) {
    T key;
    uint j;

    for (uint i = 1; i < in->end - in->start; i++) {
        key = in->start[i];
        j = i;

        while (j > 0 && in->start[j - 1] > key) {
            in->start[j] = in->start[j - 1];
            j--;
        }
        in->start[j] = key;
    }
}


template<typename T>
void merge_sort(list<T> *const in, list<T> *const out) {
    const uint n1 = in[0].end - in[0].start;
    const uint n2 = in[1].end - in[1].start;

    T L[n1];
    T R[n2];

    std::copy(in[0].start, in[0].end, &L[0]);
    std::copy(in[1].start, in[1].end, &R[0]);

    uint i = 0, l = 0, r = 0;

    out->start = in[0].start;
    out->end = in[1].end;

    // Merge-sort both lists together:
    while (l < n1 && r < n2) {
        if (R[r] < L[l])
            out->start[i++] = R[r++];
        else
            out->start[i++] = L[l++];
    }

    const uint l_rem = n1 - l;
    const uint r_rem = n2 - r;

    // Copy any remaining list elements:
    std::copy(&L[l], &L[l+l_rem], &out->start[i]);
    std::copy(&R[r], &R[r+r_rem], &out->start[i+l_rem]);
}


template<typename T>
void merge_sort(T *const start, T *const end) {
    list<T> in(start, end);

#define degree 2
    divide_and_conquer<
        list<T>,
        degree,
        SKEL_MERGE_SORT_PARALLELISATION_DEPTH,
        // Our "muscle" functions:
        is_indivisible<T>,
        divide<T, degree>,
        insertion_sort<T>,
        merge_sort<T>>(&in, 0);

#undef degree
}

} // skel

#endif // MSC_THESIS_EXERCISES_TEMPLATES_SKEL_H_
