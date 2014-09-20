#ifndef MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_
#define MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_

#include <vector>
#include <iostream>
#include <assert.h>
#include <cstdio>

#include "timer.h"

#define DEFAULT_LIST_SIZE 100000

typedef std::vector<int>  List;
typedef std::vector<List> Lists;


// Get a list populated with random values
List get_rand_list(size_t size = DEFAULT_LIST_SIZE);

// Same as get_rand_list(), but with the additional guarantee that the
// list will *not* be sorted.
List get_unsorted_list(size_t size = DEFAULT_LIST_SIZE);

// Print list contents. If "truncate" is true, print only the first 10
// elements.
void print_list(List list, bool truncate = true);

// Check is list is sorted. If "quiet" is false, print the index of
// the first unsorted element.
bool list_is_sorted(List list, bool quiet = false);


/*
 * Sorting class test templates. These functions accept a class "T"
 * which has a constructor which accepts a List, and has a function
 * get() which returns a sorted list.
 */

// Test a given list size, and print elapsed time.
template<class T>
void test_list_sort(size_t size) {
    List list = get_unsorted_list(size);

    // Start of timed section
    Timer timer;
    T sort(list);
    List sorted_list = sort.get();
    unsigned int elapsed = timer.ms();
    // End of timed section

    assert(list_is_sorted(sorted_list));

    printf("Time to sort %6lu integers: %4u ms\n", list.size(), elapsed);
}

// Test a range of list sizes, printing elapsed times.
template<class T>
void test_list_sort() {
    int i;
    size_t j;

    for (i = 0, j = 25000; i < 8; i++, j += 25000)
        test_list_sort<T>(j);
}

#endif // MSC_THESIS_EXERCISES_TEMPLATES_LIST_H_
