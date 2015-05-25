import itertools

import labm8
from labm8 import fs

ROOT = fs.path("~/src/msc-thesis")

SIMPLEBIG_BUILD = fs.path(ROOT, "skelcl/build/examples/SimpleBig")
SIMPLEBIG_BUILD_BIN = fs.path(SIMPLEBIG_BUILD, "SimpleBig")

SIMPLEBIG_SRC = fs.path(ROOT, "skelcl/examples/SimpleBig")
SIMPLEBIG_SRC_HOST = fs.path(SIMPLEBIG_SRC, "main.cpp")

DATABASE_ROOT = fs.path(ROOT, "data")
DATABASES = [
    fs.path(DATABASE_ROOT, "omnitune.skelcl.cec.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.florence.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.monza.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.tim.db"),
    fs.path(DATABASE_ROOT, "omnitune.skelcl.whz5.db")
]

BORDERS = (
    ( 1,  1,  1,  1),
    ( 5,  5,  5,  5),
    (10, 10, 10, 10),
    (20, 20, 20, 20),
    (30, 30, 30, 30),
    ( 1, 10, 30, 30),
    (20, 10, 20, 10)
)

COMPLEXITIES = ([""], ["-c"])
DATASIZES = (
    ["-w", "512", "-h", "512"],
    ["-w", "1024", "-h", "1024"],
    ["-w", "2048", "-h", "2048"],
    ["-w", "4096", "-h", "4096"]
)

ARGS = list(itertools.product(COMPLEXITIES, DATASIZES))
