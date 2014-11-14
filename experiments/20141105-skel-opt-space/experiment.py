#!/usr/bin/env python2.7
import os
import sys
from datetime import datetime

if len(sys.argv) != 2:
    print("Usage: experiment.py <target>")
    sys.exit(1)

target = sys.argv[1]
split_threshold = [10, 2000, 25]
max_recursion_depth = [0, 10, 1]
test_size = [3, 6, 1]

variable_a = split_threshold
variable_b = max_recursion_depth
variable_c = test_size

num_of_runs = 10

def prepare_experiment(split_threshold, parallelisation_depth, test_size):
    os.system(("sed -ri 's/("
               "DAC_SKEL_PARALLELISATION_DEPTH"
               "\\s+)[0-9]+/\\1{0}/' skel.h").format(parallelisation_depth))
    
    os.system(("sed -ri 's/("
               "SKEL_MERGE_SORT_SPLIT_THRESHOLD"
               "\\s+)[0-9]+/\\1{0}/' skel-merge-sort.h").format(split_threshold))

    os.system(("sed -ri 's/("
               "TEST_SIZE"
               "\\s+1e)[0-9]+/\\1{0}/' {1}.cc").format(test_size, target))
    
    os.system("make {0} >/dev/null".format(target))

def run_experiment(a, b, c):
    def get_run_time():
        start = datetime.now()
        os.system("./{0} >/dev/null".format(target))
        end = datetime.now()
        return end - start

    results = []
    for i in range(num_of_runs):
        results.append(get_run_time().microseconds)

    print a, b, 10 ** c, sum(results) / float(len(results))

print "split_threshold", "max_recursion_depth", "n", "time"
for c in range(variable_c[0], variable_c[1] + variable_c[2], variable_c[2]):
    for a in range(variable_a[0], variable_a[1] + variable_a[2], variable_a[2]):
        for b in range(variable_b[0], variable_b[1] + variable_b[2], variable_b[2]):
            prepare_experiment(a, b, c)
            run_experiment(a, b, c)
