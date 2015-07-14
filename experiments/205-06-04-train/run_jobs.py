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

errlog = open("jobsfailed.txt", "wa")
runlog = open("jobscomplete.txt", "wa")

def run_job(i, n, wgsize, program, args):
    wg_c, wg_r = unhash_params(wgsize)

    # Set environment variable.
    os.environ["OMNITUNE_OFFLINE_TRAINING"] = "1"
    os.environ["OMNITUNE_STENCIL_WG_C"] = str(wg_c)
    os.environ["OMNITUNE_STENCIL_WG_R"] = str(wg_r)

    fs.cd(fs.path(experiment.EXAMPLES_BUILD, program))

    cmd_str = "./{} {}".format(program, args.rstrip())
    cmd = cmd_str.split()

    io.info(i, "of", n, " - ", wgsize, "COMMAND:", io.colourise(io.Colours.RED, cmd_str))
    ret, _, _ = system.run(cmd, stdout=system.STDOUT, stderr=system.STDERR)

    if ret:
        print(ret, wgsize, program, args, sep="\t", file=errlog)
    else:
        print(ret, wgsize, program, args, sep="\t", file=runlog)


def get_jobs():
    joblist = "jobs/{}.txt".format(system.HOSTNAME)

    io.debug(joblist)

    if fs.isfile(joblist):
        return open(joblist).readlines()
    else:
        return []


def main():
    jobs = get_jobs()
    io.info("Loaded", len(jobs), "jobs")

    # Build example programs.
    fs.cd(experiment.EXAMPLES_BUILD)
    make.make()

    for i,job in enumerate(jobs):
        run_job(i, len(jobs), *job.split("\t"))

    lab.exit()


if __name__ == "__main__":
    main()
