# config.py - Configurable options for benchlib.
#
from os.path import basename,dirname
from sys import argv
from util import path

CWD = path(dirname(__file__))

RESULTS = path(CWD, "/results")
PLOTS = path(CWD, "/plots")

MASTER_HOSTS = ["florence", "cec"]

ID = basename(argv[0])
RUNLOG = "/tmp/{id}.run.log".format(id=ID)
