#ifndef EXERCISES_TEMPLATES_DAC_SKEL_DAC_MERGE_SORT_H_  // NOLINT(legal/copyright)
#define EXERCISES_TEMPLATES_DAC_SKEL_DAC_MERGE_SORT_H_

#include <algorithm>
#include <vector>

#include "./skel.h"

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
std::vector<ArrayType> dac_merge_sort(std::vector<ArrayType> in);

// An array length, below which the array is sorted using insertion
// sort, above which by recursive merge sort.
#define SKEL_MERGE_SORT_SPLIT_THRESHOLD 100

//
// Merge sort skeleton implementation.
////////////////////////////////////////////////////////////////////
//

// The "is_indivisble" muscle. Determine whether the list is small
// enough to sort directly (insertion sort) or to keep dividing it.
template<typename ArrayType>
bool is_indivisible(const std::vector<ArrayType>& input) {
  return input.size() <= SKEL_MERGE_SORT_SPLIT_THRESHOLD;
}


template<typename ArrayType>
std::vector<std::vector<ArrayType>>
divide(const std::vector<ArrayType>& range) {
  std::vector<std::vector<ArrayType>> out(2);

  int mid = range.size() / 2;
out[0] = std::vector<ArrayType>(range.begin(), range.begin() + mid);
out[1] = std::vector<ArrayType>(range.begin() + mid + 1, range.end());
return out;
}

// Our "conquer" muscle. A dumb insertion sort, good enough for small
// lists.
template<typename ArrayType>
std::vector<ArrayType> insertion_sort(const std::vector<ArrayType>& in) {
  std::vector<ArrayType> out = in;
  ArrayType key;
  int j;

  for (size_t i = 1; i < out.size(); i++) {
    key = out[i];
    j = i;

    while (j > 0 && out[j - 1] > key) {
      out[j] = out[j - 1];
      j--;
    }

    out[j] = key;
  }

  return out;
}


// Our "combine" muscle. Takes two sorted lists, and combines them
// into a single sorted list.
template<typename ArrayType>
std::vector<ArrayType> merge(std::vector<std::vector<ArrayType>> lists) {
  const int n1 = lists[0].size();
  const int n2 = lists[1].size();
  std::vector<ArrayType> out(n1+n2);

  // Keep the right list in-place:
  ArrayType *const L = &lists[0][0];
  ArrayType *const R = &lists[1][0];
  int i = 0, l = 0, r = 0;

  while (l < n1 && r < n2) {
    if (R[r] < L[l])
      out[i++] = R[r++];
    else
      out[i++] = L[l++];
  }

  const int l_rem = n1 - l;
  const int r_rem = n2 - r;

  // Copy any remaining list elements from left:
  std::copy(&L[l], &L[l+l_rem], &out[i]);
  std::copy(&R[r], &R[r+r_rem], &out[i + l_rem]);

  return out;
}


// Merge sort function.
template<typename ArrayType>
std::vector<ArrayType> dac_merge_sort(std::vector<ArrayType> in) {
  return divide_and_conquer<
      std::vector<ArrayType>, std::vector<ArrayType>,  // Data types
      is_indivisible<ArrayType>,                      // is_indivisible() muscle
      divide<ArrayType>,                              // divide() muscle
      insertion_sort<ArrayType>,                      // conquer() muscle
      merge<ArrayType>>                               // combine() muscle
      (in);
}

}  // namespace skel

#endif  // EXERCISES_TEMPLATES_DAC_SKEL_DAC_MERGE_SORT_H_
