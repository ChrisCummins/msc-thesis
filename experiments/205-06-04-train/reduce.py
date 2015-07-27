#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

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
from omnitune.skelcl.migrate import migrate

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
    print("Merging {n} databases:".format(n=len(dbs) + 1))
    print("   ", old_oracle)
    for db in dbs:
        print("   ", db)
    print()

    # Make a copy of the old oracle database to work from.
    io.info("Coping", old_oracle, "->", fs.basename(path))
    fs.cp(old_oracle, path)

    target = migrate(_db.Database(path=path))

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
    dbs = [migrate(_db.Database(path)) for path in
           fs.ls(experiment.DB_DEST, abspaths=True)
           if not re.search("oracle.db$", path)
           and re.search(".db$", path)]
    merge(fs.abspath(experiment.DB_DEST, "oracle.db"),
          dbs, experiment.ORACLE_PATH)


if __name__ == "__main__":
    main()
