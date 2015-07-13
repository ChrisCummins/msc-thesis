"""
Configuration.
"""
from itertools import product

import labm8
from labm8 import io
from labm8 import fs
from labm8 import system

DATA_ROOT = fs.path("~/data/msc-thesis/2015-06-23")
DB_DEST = fs.path(DATA_ROOT, "db")
ORACLE_PATH = fs.path(DATA_ROOT, "oracle.db")
TAB_ROOT = fs.path(DATA_ROOT, "tab")

# Path to classifier results using cross-validation.
CLASS_XVAL_PATH = fs.path(DATA_ROOT, "class_xval.json")
# Path to classifier results using synthetic training, real validation.
CLASS_SYN_PATH = fs.path(DATA_ROOT, "class_syn.json")

SRC_ROOT = fs.path("~/src/msc-thesis")

EXAMPLES_BUILD = fs.path(SRC_ROOT, "skelcl/build/examples/")
EXAMPLES_SRC = fs.path(SRC_ROOT, "skelcl/examples/")

#################################
# Synthetic benchmark arguments #
#################################
COMPLEXITIES = ([""], ["-c"])

BORDERS = [
    ["--north",  "1", "--south",  "1", "--east",  "1", "--west",  "1"],
    ["--north",  "5", "--south",  "5", "--east",  "5", "--west",  "5"],
    ["--north", "10", "--south", "10", "--east", "10", "--west", "10"],
    ["--north", "20", "--south", "20", "--east", "20", "--west", "20"],
    ["--north", "30", "--south", "30", "--east", "30", "--west", "30"],
    ["--north",  "1", "--south", "10", "--east", "30", "--west", "30"],
    ["--north", "20", "--south", "10", "--east", "20", "--west", "10"]
]

DATASIZES = [
    ["--width",  "512", "--height",  "512"],
    ["--width", "1024", "--height", "1024"],
    ["--width", "2048", "--height", "2048"]
]

SIMPLEBIG_ARGS = list(product(COMPLEXITIES, DATASIZES, BORDERS))

####################
# Device Arguments #
####################
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
    io.warn("Unrecognised hostname, no devargs.")
