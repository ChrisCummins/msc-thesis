set -eu

_exec() {
    ./$1 2>&1 \
        | grep -E 'Timer\[benchmark] [0-9]+' \
        | sed -r 's/.*Timer\[(.+)\] ([0-9]+).*/\1 \2/' \
        | cut -d' ' -f2
}

run() {
    cd src/
    for i in $(seq 1 10); do
        _exec mandelbrot
    done
    cd ..
}
