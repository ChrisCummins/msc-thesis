# config.py - Configurable options for benchlib.
#
from os.path import basename,dirname
from sys import argv
from util import path

CWD = path(dirname(__file__))

RESULTS = path(CWD, "/results")

_id = basename(argv[0])
RUNLOG = "/tmp/{id}.run.log".format(id=_id)
