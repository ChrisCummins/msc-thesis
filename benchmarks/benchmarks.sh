set -eu

_exec() {
    ./$1 2>&1 | \
        grep -E 'Timer\[(upload|exec|download)] [0-9]+' | \
        sed -r 's/.*Timer\[.+\] ([0-9]+).*/\1/' | \
        tr '\n' ' '
    echo
}

run() {
    cd src/
    echo "upload exec download"
    for i in $(seq 1 10); do
        _exec $1
    done
    cd ..
}
