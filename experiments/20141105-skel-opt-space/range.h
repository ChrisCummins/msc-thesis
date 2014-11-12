#ifndef EXPERIMENTS_20141105_SKEL_OPT_SPACE_RANGE_H_  // NOLINT(legal/copyright)
#define EXPERIMENTS_20141105_SKEL_OPT_SPACE_RANGE_H_

#include <vector>

//
// Storing pointers to arrays.
////////////////////////////////////////////////////////////////////
//

template<typename ArrayType>
class Range {
 public:
  ArrayType *left_;
  ArrayType *right_;

  Range() : left_(0), right_(0) {}
  Range(ArrayType *const left, ArrayType *const right) :
      left_(left), right_(right) {}
  explicit Range(const Range& src) :
      left_(src.left_), right_(src.right_) {}

  int size() const {
    return this->right_ - this->left_;
  }
};

// Split a range into multiple evenly size smaller ranges.
template<typename ArrayType, const int degree>
std::vector<Range<ArrayType>> split_range(const Range<ArrayType>& range) {
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

#endif  // EXPERIMENTS_20141105_SKEL_OPT_SPACE_RANGE_H_
