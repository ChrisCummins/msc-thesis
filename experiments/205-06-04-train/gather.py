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


def dst_path(name):
    """
    Get destination path for named database.
    """
    return fs.path(experiment.DB_DEST, name + ".db")


def cp_loc(path, name):
    """
    Copy database from local filesystem.
    """
    path = fs.path(path)
    dst = dst_path(name)

    io.info("Copying", path, "->", name)
    fs.cp(path, dst)


def cp_rmt(host, path="~/.omnitune/skelcl.db", name=None):
    """
    Copy database from remote filesystem.
    """
    name = name or host
    dst = dst_path(name)

    io.info("Copying {host}:{path}".format(host=host, path=path), "->", name)
    system.scp(host, path, dst)


def main():
    """
    Gather databases from experimental setups.
    """
    fs.mkdir(experiment.DATA_ROOT)
    fs.mkdir(experiment.DB_DEST)

    if system.HOSTNAME != "cec":
        io.fatal("script must be ran on machine `cec'")

    # TODO: Perform integrity checks. If they fail, transfer again.
    cp_loc("~/.omnitune/skelcl.db", "cec")
    cp_rmt("brendel.inf.ed.ac.uk", path="~/florence.db", name="florence")
    cp_rmt("dhcp-90-060")
    cp_rmt("monza")
    cp_rmt("tim")
    cp_rmt("whz5")

if __name__ == "__main__":
    main()
