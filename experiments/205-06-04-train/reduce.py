#!/usr/bin/env python2

from __future__ import division

import random
import sys
import os

from time import time
from datetime import datetime
from dateutil import relativedelta

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db

import experiment
import gather


def merge(dbs, path):
    """
    Merge databases into one.

    Arguments:

        dbs (list of Database): Databases to merge.
        path (str): Path to merged database.

    Returns:

        Database: merged database instance.
    """
    # Make sure that the merged database does not exist.
    fs.rm(path)
    assert not fs.isfile(path)

    target = _db.Database(path=path)

    num_runtimes = [db.num_rows("runtimes") for db in dbs]

    for db,n in zip(dbs, num_runtimes):
        io.info(("Merging {n} runtimes from {db}"
                 .format(n=n, db=fs.basename(db.path))))
        target.merge(db)

    total = target.num_rows("runtimes")

    assert total == sum(num_runtimes)

    io.info(("Merged {num_db} databases, {n} rows"
             .format(num_db=len(dbs), n=total)))

    return target


def main():
    """
    Reduce all databases to oracle.
    """
    combined_path = fs.path(experiment.DATA_ROOT, "combined.db")

    dbs = [_db.Database(path) for path in fs.ls(experiment.DB_DEST)]
    combined = merge(dbs, combined_path)

    io.info("Creating oracle ...")
    oracle = _db.MLDatabase.init_from_db(experiment.ORACLE_PATH, combined)


if __name__ == "__main__":
    main()
