#ifndef EXERCISES_TEMPLATES_DAC_RANGE_H_  // NOLINT(legal/copyright)
#define EXERCISES_TEMPLATES_DAC_RANGE_H_

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

#endif  // EXERCISES_TEMPLATES_DAC_RANGE_H_
