#!/usr/bin/env bash

set -ue

benchmark="$1"
cflags="$2"

checksum="$(./lookup.sh "$benchmark" "$cflags")"
cp -v "bin/$benchmark/$checksum" "coptbenchmarks2013/$benchmark/src/benchmark"
