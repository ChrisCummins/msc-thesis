/*
 * An implementation of integral approximation of pi using MPI, see Figure 3.4:
 * 
 *   http://books.google.co.uk/books?id=xpBZ0RyRb-oC&lpg=PP1&ots=u9fwk2OH6U&dq=Using%20MPI&pg=PA32#v=onepage&q&f=false
 */

#include <math.h>

#include "mpi.h"

#define PI 3.141592653589793238462643
#define MASTER_PROCESS (process_id == 0)

int main(int argc, char *argv[]) {

    int no_of_processes, process_id;
    int no_of_intervals = 1, i;

    double result, pi, h, sum, x, error;

    MPI::Init(argc, argv);

    no_of_processes = MPI::COMM_WORLD.Get_size();
    process_id = MPI::COMM_WORLD.Get_rank();

    while (1) {

        // Controller process
        if (MASTER_PROCESS) {
            no_of_intervals = no_of_intervals * 10 % 1000000000;
        }

        MPI::COMM_WORLD.Bcast(&no_of_intervals, 1, MPI::INT, 0);

        // Quit if we've got no intervals
        if (no_of_intervals == 0)
            break;

        h = 1.0 / (double)no_of_intervals;
        sum = 0.0;

        for (i = process_id + 1; i < no_of_intervals; i += no_of_processes) {
            x = h * ((double)i - 0.5);
            sum += (4.0 / (1.0 + x*x));
        }

        result = h * sum;

        MPI::COMM_WORLD.Reduce(&result, &pi, 1, MPI::DOUBLE, MPI::SUM, 0);

        // Print result:
        if (MASTER_PROCESS) {
            error = fabs(pi - PI);

            std::cout << "With " << no_of_intervals << " intervals, "
                      << "pi is approximately " << pi << ", "
                      << "error is " << error << "\n";
        }
    }

    MPI::Finalize();
    return 0;
}
