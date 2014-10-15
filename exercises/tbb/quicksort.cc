#include <iostream>

#include "tbb/tbb.h"

#define MIN_PARALLELISM_LENGTH 3

namespace {

template<typename T>
void quicksort_serial(T *const begin, T *const end) {
  if (end - begin > 1) {
    T *mid = std::partition(begin + 1, end, bind2nd(std::less<T>(), *begin));
    std::swap(*begin, mid[-1]);
    quicksort_serial(begin, mid - 1);
    quicksort_serial(mid, end);
  }
}

}  // namespace

template<typename T>
void quicksort(T *const begin, T *const end) {
  if (end - begin > MIN_PARALLELISM_LENGTH) {
    T *mid = std::partition(begin + 1, end, bind2nd(std::less<T>(), *begin));
    std::swap(*begin, mid[-1]);
    tbb::parallel_invoke([=]{quicksort(begin, mid - 1);},
                         [=]{quicksort(mid, end);});
  } else {
      quicksort_serial(begin, end);
  }
}

template<typename T>
void print(T *const begin, T *const end) {
  T *iterator = begin;
  while (iterator < end)
    std::cout << *iterator++ << " ";
  std::cout << "\n";
}

int main(int argc, char **argv) {
  int a[] = {2, 5, 6, 3, 1, 9, 0, 7, -4};
  const int len = sizeof(a) / sizeof(a[0]);

  print(&a[0], &a[0] + len);
  quicksort(&a[0], &a[0] + len);
  print(&a[0], &a[0] + len);

  return 0;
}
