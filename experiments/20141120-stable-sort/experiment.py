import fileinput
from itertools import product
from os import system
from re import sub
from sys import stderr

import srtime
from srtime.parser import ArgumentParser
from srtime.stats import Stats


# Integer range. Returns a list of integers in the range
# low <= i <= step.
def irange(low, high, step):
    return range(int(low), int(high + step), int(step))


# Exponential range. Iterates over the range of exponents e in
# irange(elow,ehigh,estep), returning base^e.
def erange(base, elow, ehigh, estep):
    return [base ** e for e in irange(elow, ehigh, estep)]


# Run an experiment using independent variables "variables", with
# pre-execution callback "pre_exec_cb", using timer setup
# "srtime_args".
def run(variables, pre_exec_cb, srtime_args):

    def do_run(independent_variables, args, print_header=False):
        # Run experiment with srtime.
        experiment = srtime.run(ArgumentParser().parse_args(args))

        # Collect dependent variable data.
        dependent_variables = Stats(experiment)._attrs

        # Combine results.
        results = independent_variables + dependent_variables

        # Print header if necessary.
        if print_header:
            print(", ".join([str(i[0]) for i in results]), file=stderr)

        # Print results.
        print(", ".join([str(i[1]) for i in results]), file=stderr)

    variable_names  = [v[0] for v in variables]
    variable_ranges = [v[1] for v in variables]
    print_header = True

    for values in product(*variable_ranges):
        values = list(zip(variable_names, values))
        pre_exec_cb(*[v[1] for v in values])
        do_run(values, srtime_args,
               print_header=print_header)
        print_header = False


# Our experiment
################

# Srtime arguments.
srtime_args = ["./stable_sort",
               "--filter",
               "--target-time", "0",
               "--min-iterations", "30"]

# Independent variables.
variables = [('size',      erange(10, 4, 6, 1)),
             ('threshold', irange(0, 1000, 200)),
             ('par_depth', irange(0, 5, 1))]

# User function to prepare to run an experiment.
def set_variables(size, threshold, par_depth):

    # Modify the target file to set the independent variables.
    for line in fileinput.input("stable_sort.cc", inplace=True):
        line = sub(r"(#define\s+INPUT_SIZE).+",
                   r"\1 {0}".format(size), line)
        line = sub(r"(#define\s+THRESHOLD).+",
                   r"\1 {0}".format(threshold), line)
        line = sub(r"(#define\s+PAR_DEPTH).+",
                   r"\1 {0}".format(par_depth), line)
        print(line, end="")

    # Compile code.
    system("make stable_sort")


if __name__ == "__main__":
    run(variables, set_variables, srtime_args)
