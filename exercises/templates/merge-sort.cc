#include "dc-merge-sort.h"
#include "fddc-merge-sort.h"

int main(int argc, char **argv) {

    std::cout << "DCMergeSort\n";
    test_dc_list_sort<DCMergeSort>();

    std::cout << "FDDCMergeSort\n";
    test_fddc_list_sort<FDDCMergeSort>(2);
}
