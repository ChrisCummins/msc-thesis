#!/usr/bin/env python2

from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

import labm8 as lab
from labm8 import io
from labm8 import fs
from labm8 import math as labmath

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db
from omnitune.skelcl import visualise

import experiment


def main():
    db = _db.Database(experiment.ORACLE_PATH)

    # Delete any old stuff.
    fs.rm("img")

    # Make directories
    fs.mkdir("img/coverage/devices")
    fs.mkdir("img/coverage/kernels")
    fs.mkdir("img/coverage/datasets")

    fs.mkdir("img/safety/devices")
    fs.mkdir("img/safety/kernels")
    fs.mkdir("img/safety/datasets")

    fs.mkdir("img/oracle/devices")
    fs.mkdir("img/oracle/kernels")
    fs.mkdir("img/oracle/datasets")

    # Whole-dataset plots
    visualise.sample_counts(db, "img/sample_counts.png")
    visualise.runtimes_range(db, "img/runtimes_range.png")
    visualise.max_speedups(db, "img/max_speedups.png")
    visualise.kernel_performance(db, "img/kernel_performance.png")
    visualise.device_performance(db, "img/device_performance.png")
    visualise.dataset_performance(db, "img/dataset_performance.png")
    visualise.performance_vs_coverage(db, "img/performance_vs_coverage.png")
    visualise.performance_vs_max_wgsize(db, "img/performance_vs_max_wgsize.png")
    visualise.max_wgsizes(db, "img/max_wgsizes.png")


    visualise.coverage(db, "img/coverage/coverage.png")
    visualise.safety(db, "img/safety/safety.png")
    visualise.oracle_wgsizes(db, "img/oracle/all.png")

    # Per-device plots
    for i,device in enumerate(db.devices):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        output = "img/coverage/devices/{0}.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=device)
        output = "img/oracle/devices/{0}.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where, title=device)

        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=0\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_real.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}_real.png".format(i)
        visualise.safety(db, output, where=where, title=device)
        output = "img/oracle/devices/{0}_real.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where, title=device)


        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=1\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_synthetic.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}_synthetic.png".format(i)
        visualise.safety(db, output, where=where, title=device)
        output = "img/oracle/devices/{0}_synthetic.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where, title=device)

    # Per-kernel plots
    for kernel,ids in db.lookup_named_kernels().iteritems():
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        output = "img/coverage/kernels/{0}.png".format(kernel)
        visualise.coverage(db, output=output, where=where, title=kernel)
        output = "img/safety/kernels/{0}.png".format(kernel)
        visualise.safety(db, output=output, where=where, title=kernel)
        output = "img/oracle/kernels/{0}.png".format(kernel)
        visualise.safety(db, output=output, where=where, title=kernel)

    # Per-dataset plots
    for i,dataset in enumerate(db.datasets):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        output = "img/coverage/datasets/{0}.png".format(i)
        visualise.coverage(db, output, where=where, title=dataset)
        output = "img/safety/datasets/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=dataset)
        output = "img/oracle/datasets/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=dataset)


if __name__ == "__main__":
    main()
