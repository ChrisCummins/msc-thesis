#!/usr/bin/env bash

shopt -s nullglob

suffix=$(date '+%y.%m.%d-%H.%M.%S')
tmp=.tmp
target=benchmark-reduce.log.$suffix
i=0

touch $tmp

for f in benchmark.log.*; do
    if [[ $i == 0 ]]; then
        head -n1 $f > $tmp
    fi

    echo "$f -> $target"
    cat $f | tail -n+2 >> $tmp
    i=$((i+1))
done

sort -n < $tmp > $target
rm $tmp
