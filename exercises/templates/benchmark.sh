#!/usr/bin/env bash

set -e

cpu_mult=2

echo -n "Determining number of processing units (n)... "
no_of_processors=$(nproc)
echo "$no_of_processors"

echo -n "Calculating maximum depth of parallelisation ($cpu_mult*n)... "
test_limit=$((no_of_processors*cpu_mult))
echo "$test_limit"

echo

echo "Testing std::stable_sort..."
./std-stable-sort-vector-int | tail -n+2 | awk '{print $4, $6};' > std-stable-sort-vector-int.log

echo "N std::stable_sort" > .tmp.log0
cat std-stable-sort-vector-int.log >> .tmp.log0

for (( i=0; i <= $test_limit; i++ )); do
    echo "Testing dac, parallelisation_depth = $i..."
    ./dac-int $i | tail -n+2 | awk '{print 4, $6};' > dac-int.log

    echo "dac($i)" > .tmp.log1
    awk '{print $2};' < dac-int.log >> .tmp.log1

    paste -d ' ' .tmp.log0 .tmp.log1 >> .tmp.log2
    mv .tmp.log2 .tmp.log0
done

suffix=$(date '+%y.%m.%d-%H.%M.%S')

mv .tmp.log0 benchmark.log.$suffix
rm .tmp.log1

echo
column -t benchmark.log.$suffix
