#!/usr/bin/env python
"""
Generate instruction counts and instruction count ratios for
SkelCL benchmark applications.
"""


import json
import os
import re
import subprocess
import sys

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import host

CLANG = fs.path("/home/chris/src/msc-thesis/llvm/Release+Asserts/bin/clang")
OPT = fs.path("/home/chris/src/msc-thesis/llvm/Release+Asserts/bin/opt")

def print_help():
    print("Get instruction counts for an OpenCL source file.")
    print()
    print("    Usage: %s <kernel>.cl" % sys.argv[0])
    print()
    print("Accepts a path to an OpenCL file, and compiles to LLVM bytecode.")
    print("The LLVM InstCount pass is then performed on this bytecode, and the")
    print("results printed and returned.")

def clang_cmd(opencl_input, bitcode_output):
    return [CLANG,
            "-Dcl_clang_storage_class_specifiers",
            "-isystem", "libclc/generic/include",
            "-include", "clc/clc.h",
            "-target", "nvptx64-nvidia-nvcl",
            "-xcl", "-emit-llvm" ,
            "-c", opencl_input,
            "-o", bitcode_output]

def instcount_cmd(bitcode_input):
    return [OPT, "-analyze", "-stats", "-instcount", bitcode_input]

def parse_instcount(output):
    """
    Parse the output of
    """
    line_re = re.compile("^(?P<count>\d+) instcount - Number of (?P<type>.+)")
    lines = [x.strip() for x in output.split("\n")]
    out = {}

    # Build a list of counts for each type.
    for line in lines:
        match = re.search(line_re, line)
        if match:
            count = int(match.group("count"))
            key = match.group("type")
            if key in out:
                out[key].append(count)
            else:
                out[key] = [count]

    # Sum all counts.
    for key in out:
        out[key] = sum(out[key])

    return out

def get_instcount(opencl_path):
    io.debug("Reading file '%s'" % opencl_path)

    bitcode_path = fs.path("/tmp/temp.bc")

    host.system(clang_cmd(opencl_path, bitcode_path))
    instcount_output = host.check_output(instcount_cmd(bitcode_path))
    counts = parse_instcount(instcount_output)
    return counts

def merge_counts(instcounts):
    out = {}

    for count in instcounts:
        for key in count:
            if key in out:
                out[key] += count[key]
            else:
                out[key] = count[key]
    return out

def gather():
    benchmarks = {
        "canny": {},
        "fdtd": {},
        "gol": {},
        "gaussian": {},
        "heat": {},
        "simple": {},
        "simplecomplex": {}
    }

    for benchmark in benchmarks:
        io.info("Benchmark %s" % benchmark)
        fs.cd("/home/chris/src/msc-thesis/scraps/05-12/kernels/%s" % benchmark)

        instcounts = []
        for file in fs.ls():
            instcounts.append(get_instcount(file))

        benchmarks[benchmark] = merge_counts(instcounts)

    return benchmarks

def get_ratios(data):
    out = {}

    for benchmark in data:
        ic = data[benchmark]

        total_key = "instructions (of all types)"
        total = ic[total_key]

        ratios = {}
        for key in ic:
            if key != total_key:
                name = "ratio " + key
                ratio = ic[key] / total
                ratios[name] = ratio
        ratios[total_key] = total
        out[benchmark] = ratios

    return out

def get_all_keys(data):
    allkeys = set()
    for benchmark in data:
        ic = data[benchmark]
        for key in ic:
            allkeys.add(key)
    return allkeys

def process():
    data = json.load(open("./raw.json"))

    ratios = get_ratios(data)
    allkeys = get_all_keys(ratios)

    out = {}
    for benchmark in ratios:
        ic = ratios[benchmark]
        out[benchmark] = {}
        for key in allkeys:
            if key not in ic:
                out[benchmark][key] = 0
            else:
                out[benchmark][key] = ic[key]

    io.pprint(out)


def main():
    gather()
    process()

if __name__ == "__main__":
    main()
