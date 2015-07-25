#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

import json

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


def load_config(path="~/.omnitunerc.json"):
    path = fs.abspath(path)
    if fs.isfile(path):
        return json.load(open(path))
    else:
        raise Exception("File '{}' not found!".format(path))


def main():
    """
    Push local data to remote.
    """

    cfg = load_config()
    db = migrate(_db.Database(experiment.ORACLE_PATH,
                              remote=True, remote_cfg=cfg["remote"]))
    db.push_remote()
    db.status_report()


if __name__ == "__main__":
    main()
