#!/usr/bin/env python2

from __future__ import print_function
from __future__ import division

import re
import sys

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db


def getname(source):
    lines = source.split("\n")
    for line in lines:
        match = re.search('^// "Simple" kernel', line)
        if match:
            return True, "simple"
        match = re.search('^// "Complex" kernel', line)
        if match:
            return True, "complex"
    return False, "unknown"


def create_kernel_names_table(db):
    db.execute("CREATE TABLE IF NOT EXISTS kernel_names "
               "(id text primary key,synthetic integer,name text)")

    for row in db.execute("SELECT id,source FROM kernels"):
        kernel, source = row
        synthetic, name = getname(source)

        if not db.execute("SELECT id from kernel_names where id=?",
                          (kernel,)).fetchone():
            if name == "unknown":
                print("***************** BEGIN SOURCE ***************************")
                print(source)
                name = raw_input("Name me: ")

            io.debug(kernel, synthetic, name)
            db.execute("INSERT INTO kernel_names VALUES (?,?,?)",
                       (kernel, 1 if synthetic else 0, name))
            db.commit()

def main():
    db = _db.MLDatabase("~/data/msc-thesis/oracle.db")

    create_kernel_names_table(db)

if __name__ == "__main__":
    main()
