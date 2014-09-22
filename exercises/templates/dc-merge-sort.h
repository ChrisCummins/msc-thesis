#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DC_MERGE_SORT_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DC_MERGE_SORT_H_

#include "dc.h"
#include "list.h"

/*
 * Int vector data specialization
 */
typedef DC<List> MergeSort;

template<>
bool MergeSort::isIndivisible(List list) {
    return list.size() <= 1;
}


template<>
Lists MergeSort::split(List list) {
    size_t pivot = list.size() / 2;

    // Split array in half
    List left(list.begin(), list.begin() + pivot);
    List right(list.begin() + pivot, list.end());

    List list_v[] = { left, right };
    return Lists(list_v, list_v + sizeof(list_v) / sizeof(List));
}


template<>
List MergeSort::merge(Lists lists) {
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

#endif // MSC_THESIS_EXERCISES_TEMPLATES_DC_MERGE_SORT_H_
