#!/usr/bin/env bash

set -eu

stable_sort=std-stable-sort-int
echo -n "Executing $stable_sort... "
./$stable_sort | tail -n+2 | awk '{print $2, $4};' | tr -d ',' > $stable_sort.log
echo "done"
echo "N std::stable_sort" > .tmp.log0
cat $stable_sort.log >> .tmp.log0

class_skel=merge-sort-int
echo -n "Executing $class_skel... "
./$class_skel 2 100 | tail -n+2 | awk '{print $2, $4};' | tr -d ',' > $class_skel.log
echo "done"
echo "skeleton-class" > .tmp.log1
awk '{print $2};' < $class_skel.log >> .tmp.log1
paste -d ' ' .tmp.log0 .tmp.log1 >> .tmp.log2
mv .tmp.log2 .tmp.log0

func_skel=skel-merge-sort-int
echo -n "Executing $func_skel... "
./$func_skel | tail -n+2 | awk '{print $2, $4};' | tr -d ',' > $func_skel.log
echo "done"
echo "skeleton-function" > .tmp.log1
awk '{print $2};' < $func_skel.log >> .tmp.log1
paste -d ' ' .tmp.log0 .tmp.log1 >> .tmp.log2
mv .tmp.log2 .tmp.log0

suffix=$(date '+%y.%m.%d-%H.%M.%S')

mv .tmp.log0 benchmark.log.$suffix
rm .tmp.log1

echo
column -t benchmark.log.$suffix
