#include "dac.h"

#include <assert.h>

#include "timer.h"

int main(int argc, char *argv[]) {
    List list = get_rand_list();

    Timer timer;
    MergeSort m(list);
    unsigned int elapsed = timer.ms();

    assert(list_is_sorted(m.get()));

    std::cout << "Time to sort " << list.size() << " integers: " << elapsed << " mS\n";

    return 0;
}
