#!/usr/bin/env python2

from __future__ import division

import random
import re
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


def merge(old_oracle, dbs, path):
    """
    Merge databases into one.

    Arguments:

        dbs (list of Database): Databases to merge.
        path (str): Path to merged database.

    Returns:

        Database: merged database instance.
    """
    # Make a copy of the old oracle database to work from.
    io.info("Coping", old_oracle, "->", fs.basename(path))
    fs.cp(old_oracle, path)

    target = _db.Database(path=path)

    num_runtimes = [db.num_rows("runtimes") for db in dbs]
    expected_total = target.num_rows("runtimes") + sum(num_runtimes)

    target.merge(dbs)

    total = target.num_rows("runtimes")

    if total != expected_total:
        io.fatal("Expected total", expected_total,
                 "!= actual total", total)

    io.info(("Merged {num_db} databases, {n} rows"
             .format(num_db=len(dbs), n=total)))

    return target


def main():
    """
    Reduce all databases to oracle.
    """
    dbs = [_db.Database(path) for path in
           fs.ls(experiment.DB_DEST, abspaths=True)
           if not re.search("oracle.db$", path)]
    merge(fs.abspath(experiment.DB_DEST, "oracle.db"),
          dbs, experiment.ORACLE_PATH)


if __name__ == "__main__":
    main()
