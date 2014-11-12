#ifndef EXERCISES_TEMPLATES_DAC_TEST_H_
#define EXERCISES_TEMPLATES_DAC_TEST_H_

#include <assert.h>

#include <algorithm>
#include <vector>

#include "./skel-merge-sort.h"
#include "./skel-dac-merge-sort.h"
#include "./merge-sort.h"
#include "./timer.h"
#include "./vector.h"

// Vector factories:
int           *get_unsorted_int_array(const size_t size);
vector<int>   *get_unsorted_int_vector(const size_t size);
vector<float> *get_unsorted_float_vector(const size_t size);
vector<char>  *get_unsorted_char_vector(const size_t size);

void print_result(int length, Timer *const t);

template<class T>
void test_merge_sort(vector<T> *const in,
                     const unsigned int parallelisation_depth,
                     const typename vector<T>::size_t split_threshold) {
  /*
   * "Standard" 4 phase skeleton procedure.
   */

  // Program definition:
  MergeSort<T> sort(in);

  // Parameters:
  sort.set_parallelisation_depth(parallelisation_depth);
  sort.set_split_threshold(split_threshold);

  // Execution (blocking):
  Timer t;
  sort.run();

  // Results:
  print_result(in->length, &t);
  assert(in->isSorted());

  // Free test data:
  delete[] in->data;
  delete in;
}

template<class T>
void test_sort_func(vector<T> *const in,
                    void (*sort)(T*, T*)) {
  Timer t;
  sort(in->data, in->data + in->length);
  print_result(in->length, &t);

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

  Timer t;
  std::vector<T> out = sort(vec);
  print_result(out.size(), &t);

  v.data = &out[0];
  v.length = out.size();

  assert(v.isSorted());
  delete in;
}

#endif  // EXERCISES_TEMPLATES_DAC_TEST_H_
