#include "dac.h"

#define TEST_SIZE 20

void print_list(List list) {
    for(List::iterator it = list.begin(); it != list.end(); it++)
        std::cout << *it << ", ";

    std::cout << "\n";
}

int main(int argc, char *argv[]) {
    int a[TEST_SIZE];

    for (int i = 0; i < TEST_SIZE; i++)
        a[i] = rand() % TEST_SIZE;

    List A(a, a + sizeof(a) / sizeof(int));

    print_list(A);

    MergeSort m(A);

    print_list(m.get());

    return 0;
}
