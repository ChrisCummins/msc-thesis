#!/usr/bin/env python2

import random
import sys
import os

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db

import experiment


def scp(host, src, dst):
    scp_src = "{host}:{path}".format(host=host, path=src)
    scp_dst = fs.path(dst)

    io.info("Copying", scp_src, "->", fs.basename(scp_dst), "...")

    ret,_,_ = system.run(["scp", scp_src, scp_dst],
                         stdout=system.STDOUT,
                         stderr=system.STDERR)
    if ret:
        io.error("Transfer failed!")

def main():
    """
    Gather databases from experimental setups.
    """
    fs.mkdir(experiment.DATA_ROOT)
    fs.mkdir(experiment.DB_DEST)

    # previous oracle
    oracle_src = fs.path("~/data/msc-thesis/2015-06-07/oracle.db")
    oracle_dst = fs.path(experiment.DB_DEST, "oracle.db")
    io.info("Copying", oracle_src, "->", fs.basename(oracle_dst), "...")
    fs.cp(oracle_src, oracle_dst)

    # cec
    cec_src = fs.path("~/.omnitune/skelcl.db")
    cec_dst = fs.path(experiment.DB_DEST, "cec.db")
    io.info("Copying", cec_src, "->", fs.basename(cec_dst), "...")
    fs.cp(cec_src, cec_dst)

    # dhcp-90-060
    scp("dhcp-90-060", "~/.omnitune/skelcl.db",
        fs.path(experiment.DB_DEST, "dhcp-90-060.db"))
    # florence (from brendel staging arear)
    scp("brendel.inf.ed.ac.uk", "~/florence.db",
        fs.path(experiment.DB_DEST, "florence.db"))
    # monza
    scp("monza", "~/.omnitune/skelcl.db",
        fs.path(experiment.DB_DEST, "monza.db"))
    # whz5
    scp("whz5", "~/.omnitune/skelcl.db",
        fs.path(experiment.DB_DEST, "whz5.db"))


if __name__ == "__main__":
    main()
