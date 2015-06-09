#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function


import re

import labm8 as lab
from labm8 import fs
from labm8 import io
from labm8 import latex
from labm8 import math as labmath

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db


def create_devices_table(db, output=open("gen/tables/devices.tex", "wb")):
    infos = set()
    for row in db.execute("SELECT name,count,max_compute_units,"
                          "max_clock_frequency,local_mem_size,"
                          "global_mem_size,max_work_group_size "
                          "FROM device_features"):
        name,count,cunits,freq,lmem,gmem,max_wg = row
        infos.add((count, name.strip(), cunits, freq,
                   labmath.ceil(lmem / 1024), labmath.ceil(gmem / 1024 / 1024),
                   max_wg))

    infos = list(sorted(infos, key=lambda x: x[1]))
    latex.write_table_body(infos, output=output,
                           headers=(
                               "Device count",
                               "Name",
                               "Compute units",
                               "Frequency (Hz)",
                               "Local Memory Size (KB)",
                               "Global Memory Size (MB)",
                               "Max workgroup size"
                           ))


def create_kernels_table(db, output=open("gen/tables/kernels.tex", "wb")):
    def _process_row(row):
        def _process_kernel(kernel):
            north,south,east,west = db.execute("SELECT north,south,east,west "
                                               "FROM kernels WHERE id=?",
                                               (kernel,)).fetchone()
            instcount = db.execute("SELECT instruction_count FROM "
                                   "kernel_features where id=?",
                                   (kernel,)).fetchone()[0]
            return name, north, south, east, west, instcount

        name = row[0]
        kernels = db.execute("SELECT id from kernel_names where name=?", (name,)).fetchall()
        return [_process_kernel(row[0]) for row in kernels]

    synthetics, real = set(), set()
    for row in db.execute("SELECT DISTINCT name FROM kernel_names WHERE synthetic=1"):
        [synthetics.add(entry) for entry in _process_row(row)]
    for row in db.execute("SELECT DISTINCT name FROM kernel_names WHERE synthetic=0"):
        [real.add(entry) for entry in _process_row(row)]

    synthetics = list(sorted(synthetics, key=lambda x: x[0]))
    real = list(sorted(real, key=lambda x: x[0]))

    latex.write_table_body(synthetics + real, output=output,
                           headers=("Name", "North", "South", "East",
                                    "West", "Instruction Count"))


def main():
    db = _db.MLDatabase("~/data/msc-thesis/oracle.db")
    create_devices_table(db)
    create_kernels_table(db)


if __name__ == "__main__":
    main()
