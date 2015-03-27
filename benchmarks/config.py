# config.py - Configurable options.
#
from os.path import basename,dirname
from sys import argv
from util import path

CWD = path(dirname(__file__))
SKELCL = path(CWD, '../skelcl')
SKELCL_BUILD = path(SKELCL, 'build')

_id = basename(argv[0])
RUNLOG = "/tmp/{id}.run.log".format(id=_id)
