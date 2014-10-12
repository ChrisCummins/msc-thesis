#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <ctime>

#include "mpi.h"

#define TIMESTAMP_BUFSIZE 64
#define TIMESTAMP_FORMAT "%d %B %Y %I:%M:%S %p"

void identify_process(int process_id) {
    static char buf[TIMESTAMP_BUFSIZE];
    const struct std::tm *tm;
    std::time_t now;

    now = std::time(NULL);
    tm = std::localtime(&now);
    std::strftime(buf, TIMESTAMP_BUFSIZE, TIMESTAMP_FORMAT, tm);

    // Print timestamp and ID:
    std::cout << "[" << buf << "] Process ID " << process_id << "\n";
}


int main(int argc, char *argv[]) {

    int no_of_processes;
    int process_id;

    MPI::Init(argc, argv);

    no_of_processes = MPI::COMM_WORLD.Get_size();
    process_id = MPI::COMM_WORLD.Get_rank();

    if (process_id == 0) {
        std::cout << "Launching " << no_of_processes << " processes.\n";
    }

    identify_process(process_id);

    MPI::Finalize();
    return 0;
};
