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

import experiment
import gather


def main():
    """
    Reduce all databases to oracle.
    """
    db = _db.Database(experiment.ORACLE_PATH)
    db.populate_kernel_names_table()
    db.commit()


if __name__ == "__main__":
    main()
