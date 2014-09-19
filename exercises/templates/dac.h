#ifndef MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_DAC_H_

#include <vector>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <iostream>
#include <stdexcept>
#include <sstream>

template<class T>
class DaC {

 public:
    DaC(T data) {
        this->data_ready = false;
        this->data = _process(data);
        this->data_ready = true;
    };

    bool isIndivisible(T);

    std::vector<T> split(T);

    T process(T data) {
        return data;
    };

    T merge(std::vector<T>);

    T get() {
        while (!this->data_ready)
            ;
        return this->data;
    };

 private:
    bool data_ready;
    T data;

    T _process(T data) {
        if (isIndivisible(data))
            return process(data);
        else {
            std::vector<T> split_data = split(data);

            for (std::vector<int>::size_type i = 0; i < split_data.size(); i++)
                split_data[i] = _process(split_data[i]);

            return merge(split_data);
        }

        return data;
    };
};


/*
 * Int vector data specialization
 */
typedef std::vector<int> List;
typedef std::vector<List> Lists;
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
List MergeSort::process(List list) {
    return list;
}

template<>
List MergeSort::merge(Lists lists) {
    List left = lists[0];
    List right = lists[1];
    List sorted;
    int i;

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
