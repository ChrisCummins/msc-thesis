// Copyright 2014 Chris Cummins

#ifndef EXERCISES_TEMPLATES_DAC_SKEL_H_
#define EXERCISES_TEMPLATES_DAC_SKEL_H_

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>  // NOLINT(readability/streams)
#include <memory>
#include <thread>    // NOLINT(build/c++11)
#include <vector>

#include "./debug.h"

namespace skel {

/*
 * Fixed Degree Divide and Conquer skeleton.
 *
 * Provides a template for writing divide and conquer solutions by
 * defining the 4 "muscle" functions:
 *
 * bool is_indivisible(const T& in)
 *
 *    Returns whether the problem "in" can be split into multiple
 *    sub-problems, or whether the problem should be conquered
 *    directly.
 *
 * std::vector<T> divide(const T& in);
 *
 *    Returns a pointer to an array of sub-problems that have are the
 *    result of dividing "in". This function is responsible for
 *    allocating the required memory on the heap.
 *
 * void conquer(const T& in)
 *
 *    Solve an indivisible problem "in" directly, and in-place.
 *
 * void combine(std::vector<T> in, T *const out)
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
 * less than the DAC_SKEL_PARALLELISATION_DEPTH constant, then
 * recursion will occur in a new thread. This means that the maximum
 * total number of threads created is coarsely determined by both the
 * degree and the parallelisation depth. For a fixed degree of 2:
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
    bool       is_indivisible(const ArrayType& in),
    std::vector<ArrayType> divide(const ArrayType& in),
    void       conquer(const ArrayType& in),
    void       combine(std::vector<ArrayType> in, ArrayType *const out)>
void divide_and_conquer(ArrayType *const in, const int depth = 0);


// The maximum depth at which to recurse in a new thread:
#define DAC_SKEL_PARALLELISATION_DEPTH 2

// Shorthand because we're lazy:
#define DAC_SKEL_TEMPLATE_PARAMETERS ArrayType, degree, \
      is_indivisible, divide, conquer, combine


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


// An array length, below which the array is sorted using insertion
// sort, above which by recursive merge sort.
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD       100


//
// Storing pointers to arrays.
////////////////////////////////////////////////////////////////////
//

template<typename ArrayType>
class Range {
 public:
  ArrayType *left_;
  ArrayType *right_;

  Range() {}
  Range(ArrayType *const left, ArrayType *const right)
        : left_(left), right_(right) {}
};


//
// Divide and Conquer skeleton implementation.
////////////////////////////////////////////////////////////////////
//

template<typename   ArrayType,
    const int  degree,
    bool       is_indivisible(const ArrayType& in),
    std::vector<ArrayType> divide(const ArrayType& in),
    void       conquer(const ArrayType& in),
    void       combine(std::vector<ArrayType> in, ArrayType *const out)>
void divide_and_conquer(ArrayType *const in, const int depth) {
// Cheeky shorthand:
#define self divide_and_conquer<DAC_SKEL_TEMPLATE_PARAMETERS>

  // Determine whether we're in a base case or recursion case:
  if (is_indivisible(*in)) {
    // If we can solve the problem directly, then do that:
    conquer(*in);

  } else {
    const int next_depth = depth + 1;

    // Split our problem into "k" sub-problems:
    std::vector<ArrayType> split = divide(*in);
// If the parallelisation depth is set greater than 0, then it means
// we are going to be recursing in parallel. Otherwise, we will
// *always* recurse sequentially. We can use the pre-processor to
// optimise for this case by only including the conditional logic in
// the case where the condition is actually needed (i.e. when we're
// operating in parallel).
#if DAC_SKEL_PARALLELISATION_DEPTH > 0

    // Recurse and solve for all sub-problems created by divide(). If
    // the depth is less than "parallelisation_depth", then we create
    // a new thread to perform the recursion in. Otherwise, we recurse
    // sequentially.
    if (depth < DAC_SKEL_PARALLELISATION_DEPTH) {
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
      for (auto sub_problem : sub_problems)
        self(&sub_problem, next_depth);
    }

#else  // DAC_SKEL_PARALLELISATION_DEPTH == 0

    // Sequential recursion:
    for (auto sub_problem : sub_problems)
      self(&sub_problem, next_depth);

#endif  // DAC_SKEL_PARALLELISATION_DEPTH

    // Merge the conquered "k" sub-problems into a solution:
    combine(split, in);
  }

#undef self
}


//
// Merge sort skeleton implementation.
////////////////////////////////////////////////////////////////////
//

template<typename ArrayType>
bool is_indivisible(const Range<ArrayType>& in) {
  return (in.right_ - in.left_) <= SKEL_MERGE_SORT_SPLIT_THRESHOLD;
}


template<typename ArrayType, const int degree>
std::vector<Range<ArrayType>> divide(const Range<ArrayType>& in) {
  std::vector<Range<ArrayType>> out(degree);

  const int input_length = in.right_ - in.left_;
  const int subproblem_length = input_length / degree;
  const int first_subproblem_length = input_length -
      (degree - 1) * subproblem_length;

  // Split "in" into "k" vectors, starting at address "out".
  out[0].left_ = in.left_;
  out[0].right_ = in.left_ + first_subproblem_length;

  for (int i = 1; i < degree; i++) {
    const int left = (i-1) * subproblem_length + first_subproblem_length;

    out[i].left_ = &in.left_[left];
    out[i].right_ = &in.left_[left] + subproblem_length;
  }

  return out;
}


template<typename ArrayType>
void insertion_sort(const Range<ArrayType>& in) {
  ArrayType key;
  int j;

  for (int i = 1; i < in.right_ - in.left_; i++) {
    key = in.left_[i];
    j = i;

    while (j > 0 && in.left_[j - 1] > key) {
      in.left_[j] = in.left_[j - 1];
      j--;
    }

    in.left_[j] = key;
  }
}


template<typename ArrayType>
void merge_sort(std::vector<Range<ArrayType>> in, Range<ArrayType> *const out) {
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
      Range<ArrayType>,                       // Data type
      degree,                                 // Fixed degree
      is_indivisible<ArrayType>,              // is_indivisible() muscle
      divide<ArrayType, degree>,              // divide() muscle
      insertion_sort<ArrayType>,              // conquer() muscle
      merge_sort<ArrayType>>                  // combine() muscle
      (&in);
#undef degree
}

#undef DAC_SKEL_TEMPLATE_PARAMETERS

}  // namespace skel

#endif  // EXERCISES_TEMPLATES_DAC_SKEL_H_
