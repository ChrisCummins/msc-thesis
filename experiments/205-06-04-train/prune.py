#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import json
import re

from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from scipy import stats

import weka
from weka.classifiers import Classifier as WekaClassifier
from weka.classifiers import FilteredClassifier as WekaFilteredClassifier
from weka.filters import Filter as WekaFilter
from weka.core.classes import Random as WekaRandom
from weka.core.dataset import Instances as WekaInstances

import labm8 as lab
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8 import text
from labm8.db import where

import omnitune
from omnitune.skelcl import db as _db
from omnitune.skelcl import hash_params
from omnitune.skelcl import unhash_params
from omnitune.skelcl.migrate import migrate

import experiment


def main():
    """
    Evaluate dataset and omnitune performance.
    """
    # Get the latest dataset from the oracle.
    db = migrate(_db.Database(experiment.ORACLE_PATH))

    backup_path = db.path + ".unprune"
    io.info("Writing backup to", backup_path)
    fs.cp(db.path, backup_path)

    # Strip scenarios for which there isn't enough unique workgroup
    # sizes.
    db.prune_min_params_per_scenario(25)
    # Strip scenarios so that there are at least a certain number of
    # safe parameters.
    db.prune_safe_params(3)


if __name__ == "__main__":
    main()
