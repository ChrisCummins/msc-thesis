#include "./test.h"

namespace {

int *get_rand_int_array(const size_t size) {
  int *const a = new int[size];

  for (size_t i = 0; i < size; i++)
    a[i] = rand() % size - size / 2;  // NOLINT(runtime/threadsafe_fn)

  return a;
}

float *get_rand_float_array(const size_t size) {
  float *const a = new float[size];

  for (size_t i = 0; i < size; i++)
    a[i] = (static_cast<float>(rand())  // NOLINT(runtime/threadsafe_fn)
            / static_cast<float>(size));

  return a;
}

template<typename T>
T *get_rand_array(const size_t size) {
  T *const a = new T[size];

  for (size_t i = 0; i < size; i++)
    a[i] = (static_cast<T>(rand())  // NOLINT(runtime/threadsafe_fn)
            / static_cast<T>(size));

  return a;
}

template<class T>
vector<T> *get_unsorted_vector(const size_t size,
                               T *(*get_array)(const size_t)) {
  T *const a = get_array(size);
  vector<T> *const v = new vector<T>;

  v->data = a;
  v->length = size;

  return v;
}
}  // namespace

int *get_unsorted_int_array(const size_t size) {
  return get_rand_int_array(size);
}

vector<int> *get_unsorted_int_vector(const size_t size) {
  return get_unsorted_vector(size, get_rand_int_array);
}

vector<float> *get_unsorted_float_vector(const size_t size) {
  return get_unsorted_vector(size, get_rand_float_array);
}

vector<char> *get_unsorted_char_vector(const size_t size) {
  return get_unsorted_vector(size, get_rand_array<char>);
}

void print_result(int length, Timer *const t) {
  printf("size: %7u, time: %4d ms\n", length, t->ms());
}
