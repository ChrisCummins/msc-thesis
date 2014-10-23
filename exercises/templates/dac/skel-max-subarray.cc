#include "./skel-max-subarray.h"

#include <stdio.h>
#include <limits.h>

#include "./test.h"

int main(int argc, char *argv[]) {
  std::cout << "skel::max_subarray<int>, "
            << "parallelisation depth = "
            << DAC_SKEL_PARALLELISATION_DEPTH << "\n";

  for (int i = 0, j = 1000000; i <= 10; i++, j += 100000) {
    int *a = get_unsorted_int_array(j);

    Timer t;
    int m = skel::max_subarray(&a[0], &a[0] + j);
    printf("size: %7u, time: %4d ms, answer: %d\n", j, t.ms(), m);
  }

  return 0;
}
