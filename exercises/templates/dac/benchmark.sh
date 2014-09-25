#!/usr/bin/env bash

set -e

cpu_mult=1.5

echo -n "Determining number of processing units (n)... "
no_of_processors=$(nproc)
echo "$no_of_processors"

echo -n "Calculating maximum depth of parallelisation ($cpu_mult*n)... "
test_limit=$(echo "$cpu_mult*$no_of_processors" | bc -l | xargs printf '%.0f')
echo "$test_limit"

echo

stable_sort=std-stable-sort-int
echo "Executing $stable_sort..."
./$stable_sort | tail -n+2 | awk '{print $4, $6};' > $stable_sort.log

echo "N $stable_sort" > .tmp.log0
cat $stable_sort.log >> .tmp.log0

for (( i=0; i <= $test_limit; i++ )); do
    test=merge-sort-int

    echo "Executing $test [$i of $test_limit]..."
    ./$test $i | tail -n+2 | awk '{print 4, $6};' > $test.log

    echo "d$i" > .tmp.log1
    awk '{print $2};' < $test.log >> .tmp.log1

    paste -d ' ' .tmp.log0 .tmp.log1 >> .tmp.log2
    mv .tmp.log2 .tmp.log0
done

suffix=$(date '+%y.%m.%d-%H.%M.%S')

mv .tmp.log0 benchmark.log.$suffix
rm .tmp.log1

echo
column -t benchmark.log.$suffix
