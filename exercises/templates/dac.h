#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <stdexcept>
#include <sstream>

#include "list.h"

/*
 * A divide and conquer template, in the style of an algorithmic
 * skeleton. The template defines a number of functions which are used
 * to implement divide and conquer behaviour. Individual template
 * specialisations can flesh out these "muscle" functions in order to
 * provide application specific logic.
 */
template<class T>
class DaC {

 public:
    DaC(T data, bool lazy_eval = false) {
        this->data = data;

        if (lazy_eval)
            this->data_status = IDLE;
        else
            _run();
    };

    bool isIndivisible(T);

    std::vector<T> split(T);

    T solve(T data) {
        return data;
    };

    T merge(std::vector<T>);

    T get() {
        _run();

        while (this->data_status != READY)
            ;
        return this->data;
    };

 private:

    enum {
        IDLE,
        PROCESSING,
        READY
    } data_status;

    T data;

    /*
     * The divide and conquer implementation.
     */
    T _dac(T data) {
        if (isIndivisible(data))
            return solve(data);
        else {
            std::vector<T> split_data = split(data);

            for (std::vector<int>::size_type i = 0; i < split_data.size(); i++)
                split_data[i] = _dac(split_data[i]);

            return merge(split_data);
        }

        return data;
    };

    void _run() {
        if (this->data_status == READY)
            return;

        this->data_status = PROCESSING;
        this->data = _dac(this->data);
        this->data_status = READY;
    }
};


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

#endif // MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
