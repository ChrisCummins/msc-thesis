#!/usr/bin/env python2.7
#
#   clinfo - Generate a JSON OpenCL device description blob.
#
# The function get_clinfo() returns a dictionary of the format:
#
# {
#   <platform-name>: [
#     {
#       <device-attr>: <val>,
#       <device-attr>: <val>,
#       ...
#     },
#     { ... },
#     ...
#   ],
#   <platform-name>: [ ... ],
#   ...
# }

from __future__ import print_function

from subprocess import check_output
from re import compile,search

CLINFO="clinfo"

######################
# clinfo output parser
######################

# Regular expressoins for token matching.
_int_re   = compile("^\d+$")
_float_re = compile("^\d+\.\d+$")
_freq_re  = compile("^(?P<val>\d+) ?mhz$")

# Parses a string value token, casting whole numbers to ints, real
# numbers to floats, and frequencies to ints (stripping the mhz
# suffix).
def _parse_token(value):
    freq = search(_freq_re, value)

    # Cast string -> int
    if search(_int_re, value):
        return int(value)
    # Cast string -> float
    elif search(_float_re, value):
        return float(value)
    # Cast string -> int (for mhz frequencies)
    elif freq:
        return int(freq.group("val"))
    # Return string.
    else:
        return value

# Tokenise the multi-line string output of clinfo into a list of
# (<attr>: <val>) tuples, with all lowercase letters, and appropriate
# numeric types where valid.
def _tokenise(output):
    # Split the output into lines, ignoring empty lines.
    lines = [line for line in output.split("\n") if line.strip()]

    # Split the lines based on the ":" delimiter, and convert to lowercase.
    tokenised = [[token.strip().lower() for token in line.split(":")]
                 for line in lines]

    # Ignore lines which don't have two tokens, i.e. aren't in the
    # "<name>: <value>" format.
    tokenised = filter(lambda x: len(x) == 2, tokenised)

    # Parse token values, and assign to tuples.
    tokenised = [(x[0], _parse_token(x[1])) for x in tokenised]

    return tokenised

# Parse the output of clinfo into a device description blob.
def parse(output):
    # Tokenise the output.
    tokens = _tokenise(output)
    # The output description.
    platforms = {}

    # Current state.
    platform = None
    device = None

    # Loop over the output and use a state to assign attr: val pairs.
    for attr,val in tokens:
        if attr == "platform name":
            # Update local state.
            platform = []
            # Update global state.
            platforms[val] = platform
        elif attr == "device type":
            # Update local state.
            device = {attr: val}
            # Update global state.
            platform.append(device)
        elif platform != None and device != None:
            # Update local state.
            device[attr] = val

    return platforms

# Generate a device description blob.
def get_clinfo():
    return parse(check_output(CLINFO))
