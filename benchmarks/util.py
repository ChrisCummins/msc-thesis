# util.py - Static utility methods.
#
# Take 'em or leave 'em.
from __future__ import print_function
from hashlib import sha1
from os.path import abspath,basename,dirname,exists
from subprocess import call

# Return the checksum of file at "path".
def checksum(path):
    return sha1(open(path, 'rb').read()).hexdigest()

# Concatenate all components into a path.
def path(*components):
    return abspath('/'.join(components))

# Run "args", redirecting stdout and stderr to "out". Returns exit
# status.
def system(args, out=None, exit_on_error=True):
    stdout = None if out == None else out
    stderr = None if out == None else out
    exitstatus = call(args, stdout=stdout, stderr=stderr) # exec
    if exitstatus and exit_on_error:
        print("fatal: '{0}'".format(colourise(Colours.RED,
                                              ' '.join(args))))
        exit(exitstatus)
    return exitstatus
