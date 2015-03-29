# util.py - Static utility methods.
#
# Take 'em or leave 'em.
from __future__ import print_function
from hashlib import sha1
from os import chdir,getcwd,listdir,makedirs
from os.path import abspath,basename,dirname,exists
from socket import gethostname
from subprocess import call

# directory history
_cdhist = [dirname(__file__)]

def pprint(data):
    return json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '))

# Concatenate all components into a path.
def path(*components):
    return abspath('/'.join(components))

# Change to directory "path".
def cd(path):
    cwd = pwd()
    apath = abspath(path)
    _cdhist.append(apath)
    if apath != cwd:
        chdir(apath)
    return apath


# Change to previous directory.
def cdpop():
    if len(_cdhist) > 1:
        _cdhist.pop() # Pop current directory
        chdir(_cdhist[-1]) # Change to last directory
        return _cdhist[-1]
    else:
        return pwd()


# Change back to the starting directory.
def cdstart():
    while len(_cdhist) > 2:
        cdpop()
    return cdpop()


# Change to the system root directory.
def cdroot():
    i, maxi = 0, 1000
    while cd("..") != "/" and i < maxi:
        i += 1
    if i == maxi:
        Exception("Unable to find root directory!")
    return pwd()

# Return the current working directory.
def pwd():
    return _cdhist[-1]

# List all files and directories in "path". If "abspaths", return
# absolute paths.
def ls(p=".", abspaths=True):
    if abspath:
        return [abspath(path(p, x)) for x in listdir(p)]
    else:
        return listdir(p)

def mkdir(path):
    try:
        makedirs(path)
    except OSError:
        pass

def hostname():
    return gethostname()

# Returns all of the lines in "file" as a list of strings, excluding
# comments (delimited by '#' symbol).
def parse(file):
    with open(file) as f:
        return [match('[^#]+', x).group(0).strip()
                for x in f.readlines() if not match('\s*#', x)]

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
    exitstatus = call(args, stdout=stdout, stderr=stderr, shell=True) # exec
    if exitstatus and exit_on_error:
        print("fatal: '{0}'".format(colourise(Colours.RED,
                                              ' '.join(args))))
        exit(exitstatus)
    return exitstatus

def colourise(colour, *args):
    return str(colour + str(args) + Colours.RESET)

#############################
# Shell escape colour codes #
#############################
class Colours:
    RESET   = '\033[0m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    BLUE    = '\033[94m'
    RED     = '\033[91m'

    @staticmethod
    def print(colour, *args, **kwargs):
        print(colour, end="", **kwargs)
        print(*args, end="", **kwargs)
        print(Colours.RESET, **kwargs)
