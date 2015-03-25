# config.py - Configurable options.
#
from os.path import basename
from sys import argv

ID = basename(argv[0])

RUNLOG = "/tmp/{id}.run.log".format(id=ID)
