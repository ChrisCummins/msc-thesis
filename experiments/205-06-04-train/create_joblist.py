#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import os
import random
import re
import sys

import labm8 as lab
from labm8 import db as _db
from labm8 import io
from labm8 import fmt
from labm8 import fs
from labm8 import make
from labm8 import system

from labm8.db import placeholders
from labm8.db import where

import omnitune
from omnitune import opencl
from omnitune.skelcl import hash_device
from omnitune.skelcl import unhash_params

import experiment
import train


def device2devargs(device):
    match = re.search("^([0-9]+)x(Intel)?", device)
    devcount = match.group(1)
    devtype = "CPU" if match.group(2) else "GPU"

    if device == "1xTahiti":
        host = "monza"
    if device == "1xGeForce GTX TITAN":
        host = "whz5"
    if device == "1xIntel(R) Core(TM) i7-3820 CPU @ 3.60GHz":
        host = "monza"
    if device == "1xIntel(R) Core(TM) i5-4570 CPU @ 3.20GHz":
        host = "cec"
    if device == "1xIntel(R) Core(TM) i5-2430M CPU @ 2.40GHz":
        host = "florence"

    return host, ["--device-type", devtype, "--device-count", devcount]


def enum_job(logs, db, scenario, kernel, north, south, east, west,
             width, height, device, wgsize, iterations):
    args = []

    if kernel == "complex" or kernel == "simple":
        program = "SimpleBig"
        args += [
            "--north", str(north),
            "--south", str(south),
            "--east", str(east),
            "--west", str(west)
        ]
        if kernel == "complex":
            args.append("-c")
    elif kernel == "gaussian":
        program = "GaussianBlur"
    else:
        program = "CannyEdgeDetection"

    host, devargs = device2devargs(device)

    cmd = [
        "--iterations", str(iterations),
    ] + args + devargs

    print(wgsize, program, " ".join(cmd), sep="\t", file=logs[host])


def main():
    db = _db.Database(fs.path("joblist.db"))
    data = [row for row in
            db.execute("SELECT device,Count(*) AS count\n"
                       "FROM jobs\n"
                       "GROUP BY device\n"
                       "ORDER BY count")]
    io.info("Job list:")
    print(fmt.table(data, columns=("Device", "Jobs")))
    print()

    jobs = [row for row in db.execute("SELECT * FROM jobs")]

    fs.mkdir("jobs")
    logs = {
        "monza": open("jobs/monza.txt", "w"),
        "whz5": open("jobs/whz5.txt", "w"),
        "monza": open("jobs/monza.txt", "w"),
        "cec": open("jobs/cec.txt", "w"),
        "florence": open("jobs/florence.txt", "w"),
    }

    for job in jobs:
        enum_job(logs, db, *job)

    lab.exit()


if __name__ == "__main__":
    main()
