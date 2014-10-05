#ifndef EXERCISES_TEMPLATES_DAC_VECTOR_H_
#define EXERCISES_TEMPLATES_DAC_VECTOR_H_

#include <algorithm>
#include <iostream>  // NOLINT(readability/streams)
#include <cstdio>
#include <vector>

/*
 * Generic homogeneous vector class, offering a more lightweight
 * (read: unsafe) collection type than std::vector.
 *
 * A pointer to a contiguous array of data is stored in the "data"
 * member, and the user _must_ ensure that "length" is set correctly
 * every time they wish to add or remove elements.
 */

template<class T>
class vector {
 public:
  typedef unsigned int size_t;

  T *data;
  vector<T>::size_t length;

  // Constructors:

  // Create an empty vector (no length, no data). User must allocate
  // data memory and set length:
  vector();

  // Create a vector of size "length" and allocate memory for
  // "data":
  explicit vector(const vector<T>::size_t length);

  // Create a vector of size "length" and set "data" to point to
  // data:
  vector(T *const data, const vector<T>::size_t length);

  // Create a vector that is a copy of "src":
  explicit vector(vector<T> *const src);
  ~vector();

  void print();
  bool isSorted(bool quiet = false);

  // Copy functions:
  void copy(vector<T> *const src);
  void copy(vector<T> *const src, size_t n);
  void copy(vector<T> *const src, size_t offset, size_t n);
};


/************************/
/* Template definitions */
/************************/

template<class T>
vector<T>::vector() {}

template<class T>
vector<T>::vector(const vector<T>::size_t length) {
  this->data = new T[length];
  this->length = length;
}


template<class T>
vector<T>::vector(T *const data, const vector<T>::size_t length) {
  this->data = data;
  this->length = length;
}


template<class T>
vector<T>::vector(vector<T> *const src) {
  copy(src);
}


template<class T>
vector<T>::~vector() {}


template<class T>
void vector<T>::print() {
  std::printf("%14p length: %2d, data: { ", this->data, this->length);

  const unsigned int max = this->length < 10 ? this->length : 10;

  for (unsigned int i = 0; i < max; i++)
    std::cout << this->data[i] << " ";

  if (this->length > max)
      std::cout << "...";
  else
      std::cout << "}";

  std::cout << "\n";
}


template<class T>
bool vector<T>::isSorted(bool quiet) {
  for (vector<T>::size_t i = 1; i < this->length; i++) {
    if (this->data[i] < this->data[i - 1]) {
      if (!quiet) {
        std::cout << "List item " << i << " is not sorted\n";
        this->print();
      }
      return false;
    }
  }
  return true;
}


/*************************/
/* Vector copy functions */
/*************************/

template<class T>
inline void vector<T>::copy(vector<T> *const src) {
  this->copy(src, 0, src->length);
}

template<class T>
inline void vector<T>::copy(vector<T> *const src, const size_t n) {
  this->copy(src, 0, n);
}

template<class T>
inline void vector<T>::copy(vector<T> *const src, const size_t offset,
                            const size_t n) {
  this->data = new T[n];
  memcpy(this->data, src->data + offset, n * sizeof(*src->data));
  this->length = n;
}

#endif  // EXERCISES_TEMPLATES_DAC_VECTOR_H_
