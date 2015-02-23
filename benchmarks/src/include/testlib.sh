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

    cmd="./$bin $__flags__ $@"
    echo $cmd
    $cmd
}

# Run _run_cmd() on script exit.
trap _run_cmd EXIT
