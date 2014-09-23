#include "fddc-merge-sort.h"

int main(int argc, char *argv[]) {
    std::cout << "FDDCMergesort<int>\n";
    test_fddc_list_sort<FDDCMergeSort>(2);

    return 0;
}
