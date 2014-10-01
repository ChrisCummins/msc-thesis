// Copyright 2014 Chris Cummins

#ifndef EXERCISES_TEMPLATES_DAC_SKEL_H_
#define EXERCISES_TEMPLATES_DAC_SKEL_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>  // NOLINT(readability/streams)
#include <thread>    // NOLINT(build/c++11)

#include "./debug.h"

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

template<typename   ArrayType,
         const int  degree,
         const int  parallelisation_depth,
         bool       is_indivisible(ArrayType *const in),
         ArrayType *divide(ArrayType *const in),
         void       conquer(ArrayType *const in),
         void       combine(ArrayType *const in, ArrayType *const out)>
         void       divide_and_conquer(ArrayType *const in, const int depth);


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
template<typename ArrayType>
void merge_sort(ArrayType *const left, ArrayType *const right);


#define SKEL_MERGE_SORT_PARALLELISATION_DEPTH 2
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD       100


/*
 * Generic data type for representing ranges within contiguous arrays.
 */
template<typename ArrayType>
class Range {
 public:
  ArrayType *left_;
  ArrayType *right_;

  Range() {}
  Range(ArrayType *const left, ArrayType *const right)
        : left_(left), right_(right) {}
};


/***********************************************/
/* Divide and Conquer Skeleton implementations */
/***********************************************/

template<typename   ArrayType,
         const int  degree,
         const int  parallelisation_depth,
         bool       is_indivisible(ArrayType *const in),
         ArrayType *divide(ArrayType *const in),
         void       conquer(ArrayType *const in),
         void       combine(ArrayType *const in, ArrayType *const out)>
         void divide_and_conquer(ArrayType *const in, const int depth) {
// Since the template syntax has rendered the parameterised name of
// this function unholy ugly, we'll do a cheeky macro def to keep
// things looking pretty and consistent (don't worry ma, we'll tidy up
// when we're done playing).
#define template_params ArrayType, degree, parallelisation_depth, \
               is_indivisible, divide, conquer, combine
#define self divide_and_conquer<template_params>

  // Determine whether we're in a base case or recursion case:
  if (is_indivisible(in)) {
    // If we can solve the problem directly, then do that:
    conquer(in);

  } else {
    const int next_depth = depth + 1;

    // Split our problem into "k" sub-problems:
    ArrayType *const split = divide(in);

    // Recurse and solve for all sub-problems created by divide(). If
    // the depth is less than "parallelisation_depth", then we create
    // a new thread to perform the recursion in. Otherwise, we recurse
    // sequentially.
    if (depth < parallelisation_depth) {
      // Even though "degree" is a template parameter (and so should
      // be determined and constant-ified at compile time), cpplint
      // still thinks that this is a variable-length array
      // declaration:
      std::thread threads[degree];  // NOLINT(runtime/arrays)

      // Create threads:
      for (int i = 0; i < degree; i++) {
        threads[i] = std::thread(self, &split[i], next_depth);
        DAC_DEBUG_PRINT(3, "Creating thread at depth " << next_depth);
      }

      // Block until threads complete:
      for (auto &thread : threads) {
        thread.join();
        DAC_DEBUG_PRINT(3, "Thread completed at depth " << next_depth);
      }

    } else {
      // Sequential execution (*yawn*):
      for (int i = 0; i < degree; i++)
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

template<typename ArrayType>
bool is_indivisible(Range<ArrayType> *const in) {
  return (in->right_ - in->left_) <= SKEL_MERGE_SORT_SPLIT_THRESHOLD;
}


template<typename ArrayType, const int degree>
Range<ArrayType> *divide(Range<ArrayType> *const in) {
  Range<ArrayType> *const out = new Range<ArrayType>[degree];

  const int input_length = in->right_ - in->left_;
  const int subproblem_length = input_length / degree;
  const int first_subproblem_length = input_length -
      (degree - 1) * subproblem_length;

  // Split "in" into "k" vectors, starting at address "out".
  out[0].left_ = in->left_;
  out[0].right_ = in->left_ + first_subproblem_length;

  for (int i = 1; i < degree; i++) {
    const int left = (i-1) * subproblem_length + first_subproblem_length;

    out[i].left_ = &in->left_[left];
    out[i].right_ = &in->left_[left] + subproblem_length;
  }

  return out;
}


template<typename ArrayType>
void insertion_sort(Range<ArrayType> *const in) {
  ArrayType key;
  int j;

  for (int i = 1; i < in->right_ - in->left_; i++) {
    key = in->left_[i];
    j = i;

    while (j > 0 && in->left_[j - 1] > key) {
      in->left_[j] = in->left_[j - 1];
      j--;
    }
    in->left_[j] = key;
  }
}


template<typename ArrayType>
void merge_sort(Range<ArrayType> *const in, Range<ArrayType> *const out) {
  const int n1 = in[0].right_ - in[0].left_;
  const int n2 = in[1].right_ - in[1].left_;

  ArrayType L[n1];
  ArrayType R[n2];

  std::copy(in[0].left_, in[0].right_, &L[0]);
  std::copy(in[1].left_, in[1].right_, &R[0]);

  int i = 0, l = 0, r = 0;

  out->left_ = in[0].left_;
  out->right_ = in[1].right_;

  // Merge-sort both lists together:
  while (l < n1 && r < n2) {
    if (R[r] < L[l])
      out->left_[i++] = R[r++];
    else
      out->left_[i++] = L[l++];
  }

  const int l_rem = n1 - l;
  const int r_rem = n2 - r;

  // Copy any remaining Range elements:
  std::copy(&L[l], &L[l+l_rem], &out->left_[i]);
  std::copy(&R[r], &R[r+r_rem], &out->left_[i+l_rem]);
}


template<typename ArrayType>
void merge_sort(ArrayType *const left, ArrayType *const right) {
  Range<ArrayType> in(left, right);

#define degree 2
  divide_and_conquer<
      Range<ArrayType>,
      degree,
      SKEL_MERGE_SORT_PARALLELISATION_DEPTH,
      // Our "muscle" functions:
      is_indivisible<ArrayType>,
      divide<ArrayType, degree>,
      insertion_sort<ArrayType>,
      merge_sort<ArrayType>>(&in, 0);

#undef degree
}

}  // namespace skel

#endif  // EXERCISES_TEMPLATES_DAC_SKEL_H_
