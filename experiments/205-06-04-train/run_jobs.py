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
    count_re = re.compile(r"^(\d+)x")
    devcount = re.search(count_re, device).group(1)
    name = re.sub(count_re, "", device)
    info = opencl.lookup_by_name(name)
    devtype = "GPU" if info["type"] == 4 else "CPU"

    return ["--device-type", devtype, "--device-count", devcount]


def run_job(db, kernel, north, south, east, west,
            width, height, device, wgsize):
    wg_c, wg_r = unhash_params(wgsize)

    # Set environment variable.
    os.environ["OMNITUNE_OFFLINE_TRAINING"] = "1"
    os.environ["OMNITUNE_STENCIL_WG_C"] = str(wg_c)
    os.environ["OMNITUNE_STENCIL_WG_R"] = str(wg_r)

    iterations = 30

    cmd = [
        "./SimpleBig",
        "--iterations", str(iterations),
        "--north", str(north),
        "--south", str(south),
        "--east", str(east),
        "--west", str(west),
    ] + device2devargs(device)

    if kernel == "complex":
        args += ["-c"]

    fs.cd(fs.path(experiment.EXAMPLES_BUILD, "SimpleBig"))

    cmd_str = " ".join(cmd)
    io.info("COMMAND:", io.colourise(io.Colours.RED, cmd_str))
    ret, _, _ = system.run(cmd, stdout=system.STDOUT, stderr=system.STDERR)

    values = (kernel, north, south, east, west, width,
              height, device, wgsize)

    if ret:
        db.execute("INSERT INTO jobs_failed VALUES " +
                   placeholders(*values), values)
    else:
        db.execute("INSERT INTO jobs_done VALUES " +
                   placeholders(*values), values)

    # Remove values from old tables.
    db.execute("DELETE FROM jobs WHERE " +
               where("kernel", "north", "south", "east",
                     "west", "width", "height", "device",
                     "params"),
               values)
    db.execute("DELETE FROM jobs_failed WHERE " +
               where("kernel", "north", "south", "east",
                     "west", "width", "height", "device",
                     "params"),
               values)

    db.commit()


def run_jobs(table_name, db, devices):
    jobs = [row for row in
            db.execute("SELECT *\n"
                       "FROM " + table_name + "\n"
                       "WHERE device IN " + placeholders(*devices),
                       devices)]

    total=len(jobs)
    io.info("Running {total} jobs ...".format(total=total))

    for i,job in enumerate(jobs):
        # Print progress message.
        io.info(io.colourise(io.Colours.GREEN,
                             "Running job {n} / {total} ({perc:.2f}%) ..."
                             .format(n=i, total=total, perc=(i / total) * 100)))
        run_job(db, *job)

def main():
    db = _db.Database(fs.path(experiment.DATA_ROOT, "joblist.db"))

    # Create jobs tables.
    if "jobs_done" not in db.tables:
        db.create_table_from("jobs_done", "jobs")
    if "jobs_failed" not in db.tables:
        db.create_table_from("jobs_failed", "jobs")

    data = [row for row in
            db.execute("SELECT device,Count(*) AS count\n"
                       "FROM jobs\n"
                       "GROUP BY device\n"
                       "ORDER BY count")]
    io.info("Job list:")
    print(fmt.table(data, columns=("Device", "Jobs")))
    print()

    devices = [hash_device(info["name"], 1) for info in
               opencl.get_devinfos()]

    # Build example programs.
    fs.cd(experiment.EXAMPLES_BUILD)
    make.make()

    run_jobs("jobs", db, devices)

    while db.num_rows("jobs_failed") > 0:
        run_jobs("jobs_failed", db, devices)

    lab.exit()

if __name__ == "__main__":
    main()
