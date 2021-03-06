#define INPUT_TYPE int
#define INPUT_SIZE 1000000
#define THRESHOLD 200
#define PAR_DEPTH 1

// Debug level.
#define DEBUG 0

// Use Intel Thread Building Blocks, or C++11 thread library.
#define USE_TBB 1

#include <sys/time.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <vector>

#if USE_TBB == 0
# include <thread>  // C++11 thread.
#else
# include "tbb/tbb.h"  // Intel TBB.
#endif

#if DEBUG > 0
# define DPRINT(x) std::cout << x;
#else
# define DPRINT(x)
#endif

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
    if (length < THRESHOLD || length < 2) {
        insertion_sort(start, end);
    } else {
        T *const middle = start + (end - start) / 2;
        const int next_depth = depth + 1;

        // If we haven't recursed deeper than PAR_DEPTH, then recurse
        // in parallel. Else, execute sequentially.
        if (depth < PAR_DEPTH) {
            DPRINT("p");

#if USE_TBB > 0
            // Intel TBB backend.
            tbb::parallel_invoke([=]{stable_sort(start, middle, next_depth);},
                                 [=]{stable_sort(middle, end, next_depth);});
#else
            // C++11 thread backend.
            std::thread threads[2];
            threads[0] = std::thread(stable_sort<T>, start, middle, next_depth);
            threads[1] = std::thread(stable_sort<T>, middle, end, next_depth);
            for (auto& thread : threads)
                thread.join();
#endif
        } else {
            DPRINT("-");
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

    DPRINT("\n");

    assert_sorted(start, end);

    std::cout << end_time - start_time << "\n";

    delete[] A;
}

int main(int argc, char *argv[]) {
    for (int i = 0; i < 10; i++)
        run_test();

    return 0;
}
