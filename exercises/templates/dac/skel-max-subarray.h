#ifndef EXERCISES_TEMPLATES_DAC_SKEL_MAX_SUBARRAY_H_  // NOLINT(legal/copyright)
#define EXERCISES_TEMPLATES_DAC_SKEL_MAX_SUBARRAY_H_

#include <algorithm>
#include <vector>

#include "./skel.h"
#include "./range.h"

namespace skel {

template<typename ArrayType>
class Problem {
 public:
  const Range<ArrayType> left;
  const Range<ArrayType> right;

  Problem() {}
  explicit Problem(const Range<ArrayType>& in) :
      left(in), right(Range<ArrayType>()) {}
  explicit Problem(const Problem& in) :
      left(in.left), right(in.right) {}
  Problem(const Range<ArrayType>& left, const Range<ArrayType>& right) :
      left(left), right(right) {}
};

template<typename ArrayType>
ArrayType max_subarray(const Range<ArrayType>& in);

//
// Max subarray skeleton implementation.
////////////////////////////////////////////////////////////////////
//

// The "is_indivisble" muscle.
template<typename ArrayType>
bool is_indivisible(const Problem<ArrayType>& in) {
  return in.right.size() || in.left.size() == 1;
}


// Our "conquer" muscle. A dumb insertion sort, good enough for small
// lists.
template<typename ArrayType>
ArrayType conquer(const Problem<ArrayType>& in) {
    if (in.right.size()) {
        ArrayType sum = 0, l = 0;
        for (int i = in.left.size() - 1; i >= 0; i--) {
            sum += in.left.left_[i];
            l = std::max(l, sum);
        }

        sum = 0;
        ArrayType r = 0;
        for (int i = 0; i < in.right.size(); i++) {
            sum += in.right.left_[i];
            r = std::max(r, sum);
        }

        return l + r;
    } else {
        return in.left.left_[0];
    }
}


template<typename ArrayType>
std::vector<Problem<ArrayType>> divide(const Problem<ArrayType>& in) {
  std::vector<Range<ArrayType>> ranges =
      split_range<ArrayType, 2>(in.left);

  std::vector<Problem<ArrayType>> out;
  out.push_back(Problem<ArrayType>(ranges[0]));
  out.push_back(Problem<ArrayType>(ranges[1]));
  out.push_back(Problem<ArrayType>(ranges[0], ranges[1]));

  return out;
}


// Our "combine" muscle. Takes two sorted lists, and combines them
// into a single sorted list.
template<typename ArrayType>
ArrayType combine(std::vector<ArrayType> in) {
  return std::max(in[0], in[1]) > in[2] ? std::max(in[0], in[1]) : in[2];
}


// Merge sort function.
template<typename ArrayType>
ArrayType max_subarray(ArrayType *const left, ArrayType *const right) {
  const Range<ArrayType> range(left, right);
  const Problem<ArrayType> problem(range);

  return divide_and_conquer<
      Problem<ArrayType>, ArrayType,  // Data types
      is_indivisible<ArrayType>,      // is_indivisible() muscle
      divide<ArrayType>,              // divide() muscle
      conquer<ArrayType>,             // conquer() muscle
      combine<ArrayType>>             // combine() muscle
      (problem);
}

}  // namespace skel

#endif  // EXERCISES_TEMPLATES_DAC_SKEL_MAX_SUBARRAY_H_
