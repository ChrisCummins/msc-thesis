set -e

# Internal state.
__flags__=""


append() {
    if [[ -z "$__flags__" ]]; then
        __flags__="$@"
    else
        __flags__="$__flags__ $@"
    fi
}

# If $1 == 1, then add flags.
append_if_1() {
    if [[ "$1" -eq 1 ]]; then
        shift
        append $@
    fi
}


# If $2 is set, then add flags.
append_option_if_set() {
    if [[ -n "$2" ]]; then
        append $@
    fi
}

### Private functions ###

_build_flags() {
    append_option_if_set               --device_type $device_type
    append_option_if_set               --device_count $device_count
    append_if_1          $check_result --check
    append_if_1          $logging      --logging
}

_run_cmd() {
    _build_flags

    # Assemble command.
    cmd="./$bin $__flags__ $@"

    # Print command.
    echo $cmd

    # Run command, piping all output to "output.log".
    set +e
    $cmd > output.log 2>&1
    ret=$?
    set -e

    # If program exited with non-zero status code, then dump output
    # and exit.
    if [[ $ret -ne 0 ]]; then
        cat output.log
        exit $ret
    fi

    # Create a "times.dat" file.
    grep -E 'Timer\[.*\] [0-9]' output.log | \
        sed -r 's/^.*Timer\[(.*)\] ([0-9]+) .*/\1 \2/' > times.dat

    # If verbose, print all the output. Else, print just the times.
    if [[ "$V" -eq 1 ]]; then
        cat output.log
    else
        column -t times.dat
    fi
}

# Run _run_cmd() on script exit.
trap _run_cmd EXIT
