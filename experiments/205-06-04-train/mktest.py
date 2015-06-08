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
from omnitune.skelcl.db import create_test_db,MLDatabase,Database

def main():
    """
    Generate a test dataset.
    """
    src = fs.path(sys.argv[1]
                  if len(sys.argv) > 1
                  else "~/src/msc-thesis/omnitune/tests/data/skelcl.db")
    dst = "test.db"

    io.info("Target", src)
    create_test_db(dst, Database(src), 10)

if __name__ == "__main__":
    main()
