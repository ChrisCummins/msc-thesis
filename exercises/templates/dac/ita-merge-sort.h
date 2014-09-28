#ifndef MSC_THESIS_EXERCISES_TEMPLATES_ITA_MERGE_SORT_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_ITA_MERGE_SORT_H_

#include <cstring>

#include "vector.h"

/*
 * A stable merge sort function, based on the pseudo-code from section
 * 2.3.1 of Introduction to Algorithms (3rd edition).
 */

namespace {
    template<class T>
    void ita_merge(T *const start, T *const mid, T *const end) {
        typedef unsigned long int index_t;

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
}

template<class T>
void ita_merge_sort(T *const start, T *const end) {

    if (start < end - 1) {
        T *const mid = (T *const)(start + (end - start) / 2);

        ita_merge_sort(start, mid);
        ita_merge_sort(mid, end);
        ita_merge(start, mid, end);
    }
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_ITA_MERGE_SORT_H_
