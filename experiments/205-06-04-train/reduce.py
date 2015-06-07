#!/usr/bin/env python2

from __future__ import division

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
import gather


def main():
    target_path = gather.get_db_path("target")
    target = Database(target_path)

    target.drop_table("runtimes_stats")
    target.create_table("runtimes_stats",
                        (("scenario",  "text"),
                         ("params",    "text"),
                         ("min",       "real"),
                         ("mean",      "real"),
                         ("max",       "real")))

    total = target.num_runtimes()
    i = 0
    query = target.execute("SELECT scenario,params FROM runtimes "
                           "GROUP BY scenario,params")
    for row in query:
        scenario=row[0]
        params=row[1]

        target.execute("INSERT INTO runtimes_stats SELECT "
                       "scenario,params,MIN(runtime),AVG(runtime),MAX(runtime) "
                       "FROM runtimes WHERE scenario=? AND params=?",
                       (scenario,params))
        i += 1
        if not i % 10:
            io.info(("Processed {i} rows ({p:.2f}%)"
                     .format(i=i, p=(i / total) * 100)))
            target.commit()
    target.commit()

if __name__ == "__main__":
    main()
