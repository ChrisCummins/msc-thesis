#ifndef EXPERIMENTS_20141105_SKEL_OPT_SPACE_TEST_H_
#define EXPERIMENTS_20141105_SKEL_OPT_SPACE_TEST_H_

#include <assert.h>

#include <algorithm>
#include <vector>

#include "./skel-merge-sort.h"
#include "./vector.h"

// Vector factories:
int           *get_unsorted_int_array(const size_t size);
vector<int>   *get_unsorted_int_vector(const size_t size);
vector<float> *get_unsorted_float_vector(const size_t size);
vector<char>  *get_unsorted_char_vector(const size_t size);

template<class T>
void test_sort_func(vector<T> *const in,
                    void (*sort)(T*, T*)) {
  sort(in->data, in->data + in->length);

  assert(in->isSorted());

  // Free test data:
  delete[] in->data;
  delete in;
}

template<class T>
void test_sort_func(vector<T> *const in,
                    std::vector<T> (*sort)(std::vector<T>)) {
  std::vector<T> vec(in->data, in->data + in->length);
  vector<T> v;

  std::vector<T> out = sort(vec);

  v.data = &out[0];
  v.length = out.size();

  assert(v.isSorted());
  delete in;
}

#endif  // EXPERIMENTS_20141105_SKEL_OPT_SPACE_TEST_H_
