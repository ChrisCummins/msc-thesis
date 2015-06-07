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
from omnitune.skelcl.db import Database

import experiment


def get_db_path(name):
    """
    Get the path to the named database.
    """
    date = "2015-06-07"
    return fs.path(("~/Dropbox/omnitune/{date}/{name}.db"
                    .format(date=date, name=name)))


def merge(path, dbs):
    """
    Merge databases into one.

    Arguments:

        path (str): Path to destination database.
        dbs (list of Database): Databases to merge.

    Returns:

        Database: merged database instance.
    """
    fs.rm(path)
    assert not fs.isfile(path)

    target = Database(path=path)

    num_runtimes = [db.num_runtimes() for db in dbs]

    for db,n in zip(dbs, num_runtimes):
        io.info(("Merging {n} runtimes from {db}"
                 .format(n=n, db=fs.basename(db.path))))
        target.merge(db)

    total = target.num_runtimes()

    assert total == sum(num_runtimes)

    io.info(("Merged {num_db} databases, {n} rows"
             .format(num_db=len(dbs), n=total)))

    return target


def main():
    hosts = [
        "cec",
        "dhcp-90-060",
        "florence",
        "tim",
        "whz5"
    ]

    paths = [get_db_path(host) for host in ["previous"] + hosts]
    dbs = sorted([Database(path) for path in paths if fs.isfile(path)])

    target = get_db_path("target")
    merge(target, dbs)


if __name__ == "__main__":
    main()
