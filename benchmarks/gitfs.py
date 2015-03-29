# gitfs.py - File system abstraction.
#
# Abstracts file IO over a git file system.
from atexit import register
from util import cd,system
from os.path import basename,dirname
from subprocess import call
import os

DISK_WRITE_THRESHOLD = 5

REMOTES={
    "origin": "master"
}

_diskreads = 0
_diskread = set()

_diskwrites = 0
_diskwritten = set()

def _commitandpush():
    global _diskwrites, _diskwritten

    # Escape if we have nothing to do.
    if not _diskwrites: return

    for file in _diskwritten:
        dir = dirname(file)
        base = basename(file)

        cd(dir)
        call(["git", "add", base])

    call(["git", "commit", "-m", "Auto-bot commit"])
    call(["git", "pull", "--rebase"])
    [call(["git", "push", remote, REMOTES[remote]]) for remote in REMOTES]

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
    print("Read '{path}'...".format(path=file.name))
    return file

#
def markwrite(file):
    global _diskwrites, _diskwritten

    _diskwrites += 1
    _diskwritten.add(file.name)
    print("Wrote '{path}'...".format(path=file.name))

    if _diskwrites >= DISK_WRITE_THRESHOLD:
        _commitandpush()

    return file
