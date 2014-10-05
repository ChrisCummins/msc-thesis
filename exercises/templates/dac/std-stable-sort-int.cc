#include <algorithm>

#include "./test.h"

int main(int argc, char *argv[]) {
  std::cout << "std::stable_sort<int>\n";
  for (int i = 0, j = 50000; i < 40; i++, j += 50000)
    test_sort_func(get_unsorted_int_vector(j), std::stable_sort);

  return 0;
}
