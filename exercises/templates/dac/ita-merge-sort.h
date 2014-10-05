#ifndef EXERCISES_TEMPLATES_DAC_ITA_MERGE_SORT_H_
#define EXERCISES_TEMPLATES_DAC_ITA_MERGE_SORT_H_

#include <algorithm>
#include <cstring>
#include <vector>

#include "./vector.h"

#define DEFAULT_BOTTOM_OUT_LENGTH 30

/*
 * A stable merge sort function, based on the pseudo-code from section
 * 2.3.1 of Introduction to Algorithms (3rd edition).
 */
template<class T>
void ita_merge_sort(T *const start, T *const end);

template<class T>
void ita_merge_sort(T *const start, T *const end, unsigned int threshold);


/****************************/
/* Template implementations */
/****************************/

template<class T>
void ita_merge(T *const start, T *const mid, T *const end) {
    typedef int index_t;

    const index_t n1 = mid - start;
    const index_t n2 = end - mid;

    // Create local copies of arrays:
    T left[n1];
    T right[n2];

    std::copy(start, mid, &left[0]);
    std::copy(mid, end, &right[0]);

    index_t l = 0, i = 0, r = 0;

    while (l < n1 && r < n2) {
        if (right[r] < left[l])
            start[i++] = right[r++];
        else
            start[i++] = left[l++];
    }

    while (r < n2)
        start[i++] = right[r++];

    while (l < n1)
        start[i++] = left[l++];
}

template<class T>
void ita_insertion_sort(T *const start, T *const end) {
    typedef int index_t;
    const index_t len = end - start;
    T key;
    index_t j;

    for (index_t i = 1; i < len; i++) {
        key = start[i];
        j = i;

        while (j > 0 && start[j-1] > key) {
            start[j] = start[j-1];
            j--;
        }
        start[j] = key;
    }
}


template<class T>
void ita_merge_sort(T *const start, T *const end) {
    ita_merge_sort<T>(start, end, DEFAULT_BOTTOM_OUT_LENGTH);
}


template<class T>
void ita_merge_sort(T *const start, T *const end,
                    const unsigned int threshold) {
    const unsigned int len = end - start;

    if (len > threshold) {
        // Recursion case:
        T *const mid = (T *const)(start + (end - start) / 2);

        ita_merge_sort(start, mid, threshold);
        ita_merge_sort(mid, end, threshold);
        ita_merge(start, mid, end);
    } else {
        // Base case:
        ita_insertion_sort(start, end);
    }
}

#endif  // EXERCISES_TEMPLATES_DAC_ITA_MERGE_SORT_H_
