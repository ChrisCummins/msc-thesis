#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

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
    Push local data to remote.
    """
    MYSQL_CONFIG = {
        "user": "chris",
        "database": "omnitune",
        "host": "dhcp-90-060"
    }

    db = migrate(_db.Database(experiment.ORACLE_PATH,
                              remote=True, remote_cfg=MYSQL_CONFIG))
    db.push_remote()
    db.status_report()


if __name__ == "__main__":
    main()
