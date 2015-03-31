# gitfs.py - File system abstraction.
#
# Abstracts file IO over a git file system.
from __future__ import print_function
from atexit import register
from os.path import basename,dirname
from subprocess import call

from util import cd,Colours,hostname

import config

_DISK_WRITE_THRESHOLD = 50

_REMOTES={
    "origin": "master"
}

_diskreads = 0
_diskread = set()

_diskwrites = 0
_diskwritten = set()

def _commitandpush():
    global _diskwrites, _diskwritten

    # Don't commit from master hosts.
    if hostname() in config.MASTER_HOSTS: return

    # Escape if we have nothing to do.
    if not _diskwrites: return

    Colours.print(Colours.GREEN,
                  "Commiting", len(_diskwritten), "files")

    for file in _diskwritten:
        cd(dirname(file))
        call(["git", "add", basename(file)])

    cd(CWD)
    call(["git", "commit", "-m", "Auto-bot commit"])
    call(["git", "pull", "--rebase"])
    [call(["git", "push", remote, _REMOTES[remote]]) for remote in _REMOTES]

    # Reset counters
    _diskwrites = 0
    _diskwritten = set()

# Register exit handler.
register(_commitandpush)

#
def markread(file):
    global _diskreads, _diskread

    _diskreads += 1
    _diskread.add(file.name)
    return file

#
def markwrite(file):
    global _diskwrites, _diskwritten

    _diskwrites += 1
    _diskwritten.add(file.name)

    if _diskwrites >= _DISK_WRITE_THRESHOLD:
        _commitandpush()

    return file
