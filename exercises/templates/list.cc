#include "list.h"

#include <cstdlib>

List get_rand_list(size_t size) {
    int a[size];

    for (size_t i = 0; i < size; i++)
        a[i] = rand() % size;

    return List(a, a + sizeof(a) / sizeof(int));
}


List get_unsorted_list(size_t size) {
    List list = get_rand_list(size);

    // Since we're populating the list with random values, there's a
    // chance that the list may already be sorted. In which case, try
    // again.
    return list_is_sorted(list, true) ? get_unsorted_list(size) : list;
}


void print_list(List list, bool truncate) {

    List::size_type limit;

    limit = truncate ? (list.size() > 10 ? 10 : list.size()) : list.size();

    for (List::size_type i = 0; i < limit; i++)
        std::cout << list[i] << ", ";

    if (limit < list.size())
        std::cout << "...";

    std::cout << "\n";
}


bool list_is_sorted(List list, bool quiet) {

    for (List::size_type i = 1; i < list.size(); i++)
        if (list[i] < list[i - 1]) {
            if (!quiet)
                std::cout << "List item " << i << " is not sorted.\n";
            return false;
        }

    return true;
}
