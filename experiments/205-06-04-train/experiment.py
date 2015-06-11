"""
Configuration.
"""
import itertools

import labm8
from labm8 import io
from labm8 import fs
from labm8 import system

DATA_ROOT = fs.path("~/data/msc-thesis/2015-06-11")
DB_DEST = fs.path(DATA_ROOT, "db")
ORACLE_PATH = fs.path(DATA_ROOT, "oracle.db")

SRC_ROOT = fs.path("~/src/msc-thesis")

EXAMPLES_BUILD = fs.path(SRC_ROOT, "skelcl/build/examples/")
EXAMPLES_SRC = fs.path(SRC_ROOT, "skelcl/examples/")

## Arguments

if system.HOSTNAME == "cec":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"]]
elif system.HOSTNAME == "dhcp-90-060":
    DEVARGS = [["--device-type", "GPU", "--device-count", "1"]]
elif system.HOSTNAME == "florence":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"]]
elif system.HOSTNAME == "monza":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "2"]]
elif system.HOSTNAME == "tim":
    DEVARGS = [["--device-type", "CPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "1"],
               ["--device-type", "GPU", "--device-count", "2"],
               ["--device-type", "GPU", "--device-count", "3"],
               ["--device-type", "GPU", "--device-count", "4"]]
elif system.HOSTNAME == "whz5":
    DEVARGS = [["--device-type", "GPU", "--device-count", "1"]]
else:
    io.fatal("Unrecognised hostname!")
