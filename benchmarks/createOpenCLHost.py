#!/usr/bin/env python2.7
#
from __future__ import print_function
from os.path import dirname
from re import search,sub

from util import *
from skelcl import *

import config
import json

cd(path(SKELCL_BUILD, 'tools'))
system(['make', 'OpenCLInfo'])

system(['./OpenCLInfo'], out=open(config.RUNLOG, 'w'))

output = [l.rstrip() for l in open(config.RUNLOG).readlines()]

platformid = -1
deviceid = -1

platforms = []

def addprop(name, key, cast):
    match = search(name + " *: *(.+)$", l)
    if match:
        platforms[platformid][deviceid][key] = cast(match.group(1))

for l in output:
    match = search("^Platform [0-9]+:$", l)
    if match:
        platformid += 1 # Increment platform counter.
        deviceid = -1 # Reset device counter.
        platforms.append([]) # Add a new platform.
        continue

    match = search("Device [0-9]+: (.+)$", l)
    if match:
        deviceid += 1 # Increment device counter.
        platforms[platformid].append({
            "name": match.group(1)
        }) # Add a new device.
        continue

    addprop("Device Version", "version", str)
    addprop("OpenCL C Version", "opencl_version", str)
    addprop("Compute Units", "compute_units", int)
    addprop("Clock Frequency", "clock_frequency", int)
    addprop("Global Memory Size", "global_memory_size", int)
    addprop("Local Memory Size", "local_memory_size", int)
    addprop("Max Work Group Size", "max_work_group_size", int)
    match = search("Max Work Item Sizes *: *(.+)$", l)
    if match:
        platforms[platformid][deviceid]["max_work_item_sizes"] = json.loads(match.group(1))


pprint(platforms)
