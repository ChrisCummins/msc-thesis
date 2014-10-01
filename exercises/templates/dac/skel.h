#ifndef EXERCISES_TEMPLATES_DAC_SKEL_H_  // NOLINT(legal/copyright)
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
 * Divide and Conquer skeleton.
 *
 * USAGE:
 *
 * Provides a template for writing divide and conquer solutions by
 * providing 4 "muscle" functions:
 *
 * bool is_indivisible(const T& problem)
 *
 *    Returns whether a problem can be split into multiple
 *    sub-problems, or whether the problem should be conquered
 *    directly.
 *
 * std::vector<T> divide(const T& problem);
 *
 *    Divides the input problem into a set of smaller sub-problems
 *    (when it can't be conquered directly).
 *
 * void conquer(const T& problem)
 *
 *    Solve an indivisible problem directly.
 *
 * void combine(std::vector<T> solutions, T *const out)
 *
 *    Combines a set of conquered problems ("solutions") into a single
 *    solution ("out").
 *
 * IMPLEMENTATION DETAILS:
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
template<typename ArrayType,
    bool is_indivisible(const ArrayType& problem),
    std::vector<ArrayType> divide(const ArrayType& problem),
    void conquer(const ArrayType& problem),
    void combine(std::vector<ArrayType> solutions, ArrayType *const out)>
void divide_and_conquer(ArrayType *const in, const int depth = 0);


// The maximum depth at which to recurse in a new thread:
#define DAC_SKEL_PARALLELISATION_DEPTH 2

// Shorthand because we're lazy:
#define DAC_SKEL_TEMPLATE_PARAMETERS \
    ArrayType, is_indivisible, divide, conquer, combine


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
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD 100


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
    bool       is_indivisible(const ArrayType& problem),
    std::vector<ArrayType> divide(const ArrayType& problem),
    void       conquer(const ArrayType& problem),
    void       combine(std::vector<ArrayType> problem, ArrayType *const out)>
void divide_and_conquer(ArrayType *const problem, const int depth) {
// Cheeky shorthand:
#define self divide_and_conquer<DAC_SKEL_TEMPLATE_PARAMETERS>

  // Determine whether we're in a base case or recursion case:
  if (is_indivisible(*problem)) {
    // If we can solve the problem directly, then do that:
    conquer(*problem);

  } else {
    const int next_depth = depth + 1;

    // Split our problem into "k" sub-problems:
    std::vector<ArrayType> sub_problems = divide(*problem);

// If the parallelisation depth is set greater than 0, then it means
// we are going to be recursing in parallel. Otherwise, we will
// *always* recurse sequentially. We can use the pre-processor to
// optimise for this case by only including the conditional logic in
// the case where the condition is actually needed (i.e. when we're
// operating in parallel).
#if DAC_SKEL_PARALLELISATION_DEPTH > 0

    // Recurse and solve for all sub-problems created by divide(). If
    // the depth is less than the parallelisation depth then we create
    // a new thread to perform the recursion in. Otherwise, we recurse
    // sequentially.
    if (depth < DAC_SKEL_PARALLELISATION_DEPTH) {
      std::vector<std::thread> threads(sub_problems.size());

      // Create threads and block until completed:
      for (size_t i = 0; i < sub_problems.size(); i++) {
        threads[i] = std::thread(self, &sub_problems[i], next_depth);
        DAC_DEBUG_PRINT(3, "Creating thread at depth " << next_depth);
      }
      for (auto& thread : threads) {
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
    combine(sub_problems, problem);
  }

#undef self
}


// The "divide" muscle. Takes a range and splits into "degree" sections.
template<typename ArrayType, const int degree>
std::vector<Range<ArrayType>> divide_range(const Range<ArrayType>& range) {
  std::vector<Range<ArrayType>> out(degree);

  const int input_length = range.right_ - range.left_;
  const int subproblem_length = input_length / degree;
  const int first_subproblem_length = input_length -
      (degree - 1) * subproblem_length;

  // Split "range" into "k" vectors, starting at address "out".
  out[0].left_ = range.left_;
  out[0].right_ = range.left_ + first_subproblem_length;

  for (int i = 1; i < degree; i++) {
    const int left = (i-1) * subproblem_length + first_subproblem_length;

    out[i].left_ = &range.left_[left];
    out[i].right_ = &range.left_[left] + subproblem_length;
  }

  return out;
}


//
// Merge sort skeleton implementation.
////////////////////////////////////////////////////////////////////
//

// The "is_indivisble" muscle. Determine whether the list is small
// enough to sort directly (insertion sort) or to keep dividing it.
template<typename ArrayType>
bool is_indivisible(const Range<ArrayType>& range) {
  return (range.right_ - range.left_) <= SKEL_MERGE_SORT_SPLIT_THRESHOLD;
}


// Our "conquer" muscle. A dumb insertion sort, good enough for small
// lists.
template<typename ArrayType>
void insertion_sort(const Range<ArrayType>& range) {
  ArrayType key;
  int j;

  for (int i = 1; i < range.right_ - range.left_; i++) {
    key = range.left_[i];
    j = i;

    while (j > 0 && range.left_[j - 1] > key) {
      range.left_[j] = range.left_[j - 1];
      j--;
    }

    range.left_[j] = key;
  }
}


// Our "combine" muscle. Takes two sorted lists, and combines them
// into a single sorted list.
template<typename ArrayType>
void merge(std::vector<Range<ArrayType>> range,
           Range<ArrayType> *const out) {
  const int n1 = range[0].right_ - range[0].left_;
  const int n2 = range[1].right_ - range[1].left_;

  ArrayType L[n1];
  ArrayType R[n2];

  std::copy(range[0].left_, range[0].right_, &L[0]);
  std::copy(range[1].left_, range[1].right_, &R[0]);

  int i = 0, l = 0, r = 0;

  out->left_ = range[0].left_;
  out->right_ = range[1].right_;

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


// Merge sort function.
template<typename ArrayType>
void merge_sort(ArrayType *const left, ArrayType *const right) {
  Range<ArrayType> range(left, right);

  divide_and_conquer<
      Range<ArrayType>,            // Data type
      is_indivisible<ArrayType>,   // is_indivisible() muscle
      divide_range<ArrayType, 2>,  // divide() muscle
      insertion_sort<ArrayType>,   // conquer() muscle
      merge<ArrayType>>            // combine() muscle
      (&range);
}

#undef DAC_SKEL_TEMPLATE_PARAMETERS

}  // namespace skel

#endif  // EXERCISES_TEMPLATES_DAC_SKEL_H_
