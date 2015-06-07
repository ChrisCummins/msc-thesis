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
from omnitune.skelcl.db import Database

import experiment
import gather


def main():
    target_path = gather.get_db_path("target")
    target = Database(target_path)

    # Create runtimes_stats table.
    target.drop_table("runtimes_stats")
    target.create_table("runtimes_stats",
                        (("scenario",    "text"),
                         ("params",      "text"),
                         ("num_samples", "integer"),
                         ("min",         "real"),
                         ("mean",        "real"),
                         ("max",         "real")))

    # Fetch all unique (scenario, params) pairs.
    query = target.execute("SELECT scenario,params FROM runtimes "
                           "GROUP BY scenario,params")
    rows = query.fetchall()
    total = len(rows)

    start_time = time()
    i = 0

    # Iterate over each row.
    for row in rows:
        scenario=row[0]
        params=row[1]

        # Gather statistics about runtimes for a (scenario, params)
        # pair.
        target.execute("INSERT INTO runtimes_stats SELECT scenario,params,"
                       "COUNT(runtime),MIN(runtime),AVG(runtime),MAX(runtime) "
                       "FROM runtimes WHERE scenario=? AND params=?",
                       (scenario,params))

        # Intermediate progress reports.
        i += 1
        if not i % 10:
            # Commit progress.
            target.commit()

            # Estimate job completion time.
            elapsed = time() - start_time
            remaining_rows = total - i
            rate = i / elapsed

            dt1 = datetime.fromtimestamp(0)
            dt2 = datetime.fromtimestamp(rate * remaining_rows)
            rd = relativedelta.relativedelta(dt2, dt1)

            io.info("Progress: {0:.3f}%. Estimated completion in "
                    "{1:02d}:{2:02d}:{3:02d}."
                     .format((i / total) * 100,
                             rd.hours, rd.minutes, rd.seconds))

    # Done.
    target.commit()

if __name__ == "__main__":
    main()
