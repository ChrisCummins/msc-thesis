#!/usr/bin/env python2

from __future__ import division

import sys

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db


def create_oracle_wgsizes_heatmaps(db):
    space = db.oracle_param_space()
    space.heatmap("img/oracle/heatmap.png",
                  title="All")

    for i,device in enumerate(db.devices):
        io.debug("Device heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE device='{0}')"
                 .format(device))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/devices/{0}.png"
                      .format(i), title=device)

    for i,dataset in enumerate(db.datasets):
        io.debug("Dataset heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE dataset='{0}')"
                 .format(dataset))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/datasets/{0}.png"
                      .format(i), title=dataset)

    for kernel,ids in db.lookup_named_kernels().iteritems():
        io.debug("Kernel heatmap", kernel, "...")
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN (SELECT id FROM scenarios WHERE "
                 "kernel IN ({0}))".format(",".join(id_wrapped)))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/kernels/{0}.png"
                      .format(kernel), title=kernel)


def create_max_wgsizes_heatmaps(db):
    space = db.max_wgsize_space()
    space.heatmap("img/max_wgsizes.png",
                  title="Distribution of maximum workgroup sizes")


def eval_static_wgsizes(db):
    for i,device in enumerate(db.devices):
        io.debug("Device coverage", i, "...")
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/devices/{0}.png"
                      .format(i), title=device)
        io.debug("Device safety", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/devices/{0}.png"
                      .format(i), title=device)

    for i,dataset in enumerate(db.datasets):
        io.debug("Dataset coverage", i, "...")
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/datasets/{0}.png"
                      .format(i), title=dataset)
        io.debug("Device safety", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/datasets/{0}.png"
                      .format(i), title=dataset)

    for kernel,ids in db.lookup_named_kernels().iteritems():
        io.debug("Kernel coverage", i, "...")
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/kernels/{0}.png"
                      .format(i), title=kernel)
        io.debug("Kernel safety", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/kernels/{0}.png"
                      .format(kernel), title=kernel)


def main():
    db = _db.MLDatabase("~/data/msc-thesis/oracle.db")

    eval_static_wgsizes(db)

    create_oracle_wgsizes_heatmaps(db)
    create_max_wgsizes_heatmaps(db)


if __name__ == "__main__":
    main()
