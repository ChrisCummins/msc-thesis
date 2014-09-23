#ifndef MSC_THESIS_EXERCISES_TEMPLATES_FDDC_MERGE_SORT_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_FDDC_MERGE_SORT_H_

#include "fddc.h"
#include "list.h"

/*
 * List specialization
 */
typedef FDDC<List> FDDCMergeSort;

template<>
bool FDDCMergeSort::isIndivisible(List list) {
    return list.size() <= 1;
}


template<>
List *FDDCMergeSort::split(List list) {
    size_t pivot = list.size() / 2;

    // Split array in half
    List left(list.begin(), list.begin() + pivot);
    List right(list.begin() + pivot, list.end());

    List *split = new List[2];
    split[0] = left;
    split[1] = right;

    return split;
}


template<>
List FDDCMergeSort::merge(List *lists) {
    List left = lists[0];
    List right = lists[1];
    List sorted = List(left.size() + right.size());

    const List::size_type left_size  = left.size();
    const List::size_type right_size = right.size();

    // Iterators:
    List::size_type l = 0, r = 0, i = 0;

    while (l < left_size && r < right_size) {
        if (right[r] < left[l])
            sorted[i++] = right[r++];
        else
            sorted[i++] = left[l++];
    }

    while (r < right_size)
        sorted[i++] = right[r++];

    while (l < left_size)
        sorted[i++] = left[l++];

    return sorted;
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_FDDC_MERGE_SORT_H_
