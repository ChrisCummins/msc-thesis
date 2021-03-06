#ifndef EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_MERGE_SORT_H_  // NOLINT(legal/copyright)
#define EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_MERGE_SORT_H_

#include <algorithm>
#include <vector>

#include "./skel.h"
#include "./range.h"

namespace skel {


// A concrete stable merge sort implementation, using the Divide and
// Conquer skeleton.
//
// The only muscle function we need to provide is merge(), since the
// defaults for the FDDC template are satisfactory.
//
// MergeSort requires that the class "T" supports comparion and
// equality operators.
//
template<typename ArrayType>
void merge_sort(ArrayType *const left, ArrayType *const right);

// An array length, below which the array is sorted using insertion
// sort, above which by recursive merge sort.
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD 35

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

  // Make a copy of the left list on stack:
  ArrayType L[n1];
  std::copy(range[0].left_, range[0].right_, &L[0]);

  // Keep the right list in-place:
  ArrayType *const R = range[1].left_;

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

  // Copy any remaining range elements from left:
  std::copy(&L[l], &L[l+l_rem], &out->left_[i]);
}


// Merge sort function.
template<typename ArrayType>
void merge_sort(ArrayType *const left, ArrayType *const right) {
  // Wrap the input parameters into a Range object, for passing to the
  // divide-and-transform skeleton. We need to use a wrapper object
  // since the divide-and-transform skeleton expects only a single
  // input argument.
  Range<ArrayType> range(left, right);

  // Instantiate the divide-and-transform (i.e. in-place
  // divide-and-conquer) skeleton with our wrapped data, and our
  // muscle functions.
  divide_and_transform<
      Range<ArrayType>,           // Data type
      is_indivisible<ArrayType>,  // Is-indivisible muscle
      split_range<ArrayType, 2>,  // Divide muscle
      insertion_sort<ArrayType>,  // Conquer muscle
      merge<ArrayType>>           // Combine muscle
      (&range);                   // Wrapped data
}

}  // namespace skel

#endif  // EXPERIMENTS_20141105_SKEL_OPT_SPACE_SKEL_MERGE_SORT_H_
