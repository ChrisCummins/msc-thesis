#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "tbb/tbb.h"

#define INPUT_TYPE int
#define INPUT_SIZE 1e6
#define THRESHOLD 100
#define PAR_DEPTH 2

// Debug level.
#define DEBUG 0

template<typename T>
void insertion_sort(T *const start, T *const end);

template<typename T>
void merge(T *const start, T *const middle, T *const end);

template<typename T>
void stable_sort(T *const start, T *const end, int depth = 0) {
    const int length = end - start;

    // If our array contains fewer elements than the THRESHOLD value,
    // then sort it directly using an insertion sort. Otherwise, split
    // the array in half and solve recursively, before merging the two
    // halves.
    if (length < THRESHOLD) {
        insertion_sort(start, end);
    } else {
        T *const middle = start + (end - start) / 2;
        const int next_depth = depth + 1;

        // If we haven't recursed deeper than PAR_DEPTH, then recurse
        // in parallel. Else, execute sequentially.
        if (depth < PAR_DEPTH) {
#if DEBUG > 0
            std::cout << "p";
#endif
            tbb::parallel_invoke([=]{stable_sort(start, middle, next_depth);},
                                 [=]{stable_sort(middle, end, next_depth);});
        } else {
#if DEBUG > 0
            std::cout << "-";
#endif
            stable_sort(start, middle, next_depth);
            stable_sort(middle, end, next_depth);
        }

        merge(start, middle, end);
    }
}

template<typename T>
void insertion_sort(T *const start, T *const end) {
    T key;
    int j;
    const int length = end - start;

    for (int i = 1; i < length; i++) {
        key = start[i];
        j = i;

        while (j > 0 && start[j - 1] > key) {
            start[j] = start[j - 1];
            j--;
        }

        start[j] = key;
    }
}

template<typename T>
void merge(T *const start, T *const middle, T *const end) {
    const int n1 = middle - start;
    const int n2 = end - middle;

    std::vector<T> buf = std::vector<T>(start, middle);

    T *const L = &buf[0];
    T *const R = middle;

    int i = 0, l = 0, r = 0;

    while (l < n1 && r < n2) {
        if (R[r] < L[l])
            start[i++] = R[r++];
        else
            start[i++] = L[l++];
    }

    const int l_rem = n1 - l;
    std::copy(&L[l], &L[l + l_rem], &start[i]);
}

template<typename T>
void print(T *const start, T *const end) {
    T *i = start;

    while (i < end) {
        std::cout << *i << " ";
        i++;
    }

    std::cout << "\n";
}

template<typename T>
T *rand_array(const size_t size) {
    T *const a = new T[size];

    for (size_t i = 0; i < size; i++)
        a[i] = static_cast<T>(rand());  // NOLINT(runtime/threadsafe_fn)

    return a;
}

template<typename T>
void assert_sorted(T *const start, T *const end) {
    T *i = start + 1;

    while (i < end) {
        assert(*i >= *(i - 1));
        i++;
    }
}

int64_t now() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec * 1000 + tp.tv_usec / 1000;
}

void run_test() {
    int n = INPUT_SIZE;
    INPUT_TYPE *A = rand_array<INPUT_TYPE>(n);

    INPUT_TYPE *const start = &A[0];
    INPUT_TYPE *const end = &A[n];

    int64_t start_time = now();
    stable_sort(start, end);
    int64_t end_time = now();

#if DEBUG > 0
    std::cout << "\n";
#endif

    assert_sorted(start, end);

    std::cout << end_time - start_time << "\n";

    delete[] A;
}

int main(int argc, char *argv[]) {
    for (int i = 0; i < 10; i++)
        run_test();

    return 0;
}
