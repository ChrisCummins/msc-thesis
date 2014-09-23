#include "dc-merge-sort.h"

int main(int argc, char *argv[]) {
    std::cout << "DCMergesort<int>\n";
    test_dc_list_sort<DCMergeSort>();

    return 0;
}
