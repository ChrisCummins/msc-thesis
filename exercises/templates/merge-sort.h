#ifndef MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_

#include "dac.h"
#include "list.h"

/*
 * Int vector data specialization
 */
typedef DaC<List> MergeSort;

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
List MergeSort::solve(List list) {
    return list;
}


template<>
List MergeSort::merge(Lists lists) {
    List left = lists[0];
    List right = lists[1];
    List sorted;

    while (left.size() || right.size()) {
        if (left.size() && right.size()) {
            if (right.front() < left.front()) {
                sorted.push_back(right.front());
                right.erase(right.begin());
            } else {
                sorted.push_back(left.front());
                left.erase(left.begin());
            }
        } else {

            while (right.size()) {
                sorted.push_back(right.front());
                right.erase(right.begin());
            }

            while (left.size()) {
                sorted.push_back(left.front());
                left.erase(left.begin());
            }
        }
    }

    return sorted;
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_MERGE_SORT_H_
