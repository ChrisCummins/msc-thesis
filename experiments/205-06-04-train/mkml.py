#!/usr/bin/env python2

import sys

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl
from omnitune.skelcl.db import MLDatabase,Database

import experiment
import gather


def main():
    src = fs.path(sys.argv[1]
                  if len(sys.argv) > 1
                  else "~/src/msc-thesis/omnitune/tests/data/skelcl.db")
    dst = "ml.db"

    io.info("Target", src)
    MLDatabase.init_from_db(dst, Database(src))


if __name__ == "__main__":
    main()
