import itertools

import labm8
from labm8 import io
from labm8 import fs
from labm8 import system


ROOT = fs.path("~/src/msc-thesis")

SIMPLEBIG_BUILD = fs.path(ROOT, "skelcl/build/examples/SimpleBig")
SIMPLEBIG_BUILD_BIN = fs.path(SIMPLEBIG_BUILD, "SimpleBig")

SIMPLEBIG_SRC = fs.path(ROOT, "skelcl/examples/SimpleBig")
SIMPLEBIG_SRC_HOST = fs.path(SIMPLEBIG_SRC, "main.cpp")

DATABASE_ROOT = fs.path(ROOT, "data")
DATABASES = [
    fs.path(DATABASE_ROOT, "omnitune.skelcl.cec.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.dhcp-90-060.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.florence.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.monza.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.tim.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.whz5.db")
]
DATABASE_ORACLE = "omnitune.oracle.db"

## Arguments

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

ARGS = list(itertools.product(COMPLEXITIES, DATASIZES, BORDERS, DEVARGS))
