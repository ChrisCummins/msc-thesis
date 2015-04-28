#!/usr/bin/env python2.7
#
#   hwinfo - Generate a JSON hardware description blob.
#
# The function get_clinfo() returns a dictionary of the format:
#
# {
#   <hostname>: {
#     "host": {
#       <host-attr>: <val>,
#       <host-attr>: <val>,
#       ...
#     }
#     "platforms": {
#       <platform-name>: [
#         {
#           <device-attr>: <val>,
#           <device-attr>: <val>,
#           ...
#         },
#         { ... },
#         ...
#       ],
#       <platform-name>: [ ... ],
#       ...
#    },
#    ...
# }

from __future__ import print_function

from clinfo import get_clinfo
from json import dumps
from re import compile,search,sub
from subprocess import check_output
from util import hostname,pprint

# CPU max frequency file path.
CPU_MAX_FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq"

# Get the processor frequency.
def get_freq():
    return int(open(CPU_MAX_FREQ).readlines()[0].strip())

# Memory info file path.
MEM_INFO = "/proc/meminfo"

# Get the system memory, in kb.
def get_mem():
    return int(open(MEM_INFO).readlines()[0].split()[1])

# Nproc command.
NPROC = "nproc"

# Return the number of processors.
def get_nproc():
    return int(check_output(NPROC))

# CPU info file path.
CPUINFO = "/proc/cpuinfo"

# Model name regex.
model_name_re = compile("^[ \t]*model name[ \t]*: +")

# Return the name of the processor (assumes a single processor).
def get_procname():
    # Iterate over the lines in the cpuinfo file.
    for line in open(CPUINFO).readlines():
        # If line contains model name, return it.
        if search(model_name_re, line):
            return sub(model_name_re, "", line).strip()

# Return a JSON hardware description blob.
def get_hwinfo():
    return {
        hostname(): {
            "host": {
                "procname": get_procname(),
                "nproc": get_nproc(),
                "freq": get_freq(),
                "mem": get_mem()
            },
            "platforms": get_clinfo()
        }
    }

pprint(get_hwinfo())
