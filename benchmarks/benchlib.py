from __future__ import print_function
from datetime import datetime
from hashlib import sha1
from math import sqrt
from os import chdir,getcwd
from os.path import abspath,dirname
from re import match
from subprocess import call
from sys import exit,stdout


##### LOCAL VARIABLES #####

# directory history
__cdhist = [dirname(__file__)]


##### UTILITIES #####

# Concatenate all components into a path.
def path(*components):
    return abspath('/'.join(components))

# Return the path to binary directory of example program "name".
def bindir(name):
    return path(SKELCL_BUILD, 'examples', name)

# Return the path to binary file of example program "name".
def bin(name):
    return path(bindir(name), name)

# Change to directory "path".
def cd(path):
    cwd = pwd()
    apath = abspath(path)
    __cdhist.append(apath)
    if apath != cwd:
        chdir(apath)
    return apath


# Change to previous directory.
def cdpop():
    if len(__cdhist) > 1:
        __cdhist.pop() # Pop current directory
        chdir(__cdhist[-1]) # Change to last directory
        return __cdhist[-1]
    else:
        return pwd()


# Change back to the starting directory.
def cdstart():
    while len(__cdhist) > 2:
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
    return __cdhist[-1]

# Returns all of the lines in "file" as a list of strings, excluding
# comments (delimited by '#' symbol).
def parse(file):
    with open(file) as f:
        return [match('[^#]+', x).group(0).strip() for x in f.readlines() if not match('\s*#', x)]

# Return the current date in style "format".
def datestr(format="%I:%M%p on %B %d, %Y"):
    return datetime.now().strftime(format)

def printheader(file=stdout):
    print('{0} in {1}'.format(datestr(), getcwd()), file=file)
    file.flush()

# Run "args", redirecting stdout and stderr to "out". Returns exit
# status.
def system(args, out=None, exit_on_error=True):
    stdout = None if out == None else out
    stderr = None if out == None else out
    exitstatus = call(args, stdout=stdout, stderr=stderr) # exec
    if exitstatus and exit_on_error:
        print("fatal: '{0}'".format(' '.join(args)))
        exit(exitstatus)
    return exitstatus


# Returns the secure checksum of "file".
def checksum(file):
    return sha1(open(file).read()).hexdigest()


####### STATS #######

# Return the mean value of a list of divisible numbers.
def mean(num):
    if len(num):
        return sum(num) / len(num)
    else:
        return 0

# Return the variance of a list of divisible numbers.
def variance(num):
    if len(num) > 1:
        m = mean(num)
        return sum([(x - m) ** 2 for x in num]) / (len(num) - 1)
    else:
        return 0

# Return the standard deviation of a list of divisible numbers.
def stdev(num):
    return sqrt(variance(num))


####### CONSTANTS & CONFIG #######

CWD = path(dirname(__file__))
SKELCL = path(CWD, '../skelcl')
SKELCL_BUILD = path(SKELCL, 'build')
BUILDLOG = path(CWD, 'make.log')
RUNLOG = path(CWD, 'run.log')


###### BENCHMARK FUNCTIONS #####

# Build example program "prog". If "clean", then clean before
# building.
#
#   @side-effect: Changes working dir.
def make(prog, clean=True):
    progdir, progbin = bindir(prog), bin(prog)

    cd(progdir)
    with open(BUILDLOG, 'w') as f:
        printheader(f)
        if clean:
            system(['make', 'clean'], out=f)
        system(['make', prog], out=f)

# Build SkelCL. If "configure", run cmake.
#
#   @side-effect: Changes working dir.
def makeSkelCL(configure=True, clean=True):
    cd(SKELCL_BUILD)
    with open(BUILDLOG, 'w') as f:
        printheader(f)
        if configure:
            system(['cmake', '..'], out=f)
        if clean:
            system(['make', 'clean'], out=f)
        system(['make'], out=f)

# Run program "prog" and return a dictionary of runtimes.
#
#   @side-effect: Changes working dir.
def time(prog):
    progdir, progbin = bindir(prog), bin(prog)

    cd(progdir)
    with open(RUNLOG, 'w') as f:
        printheader(f)
        system([progbin], out=f)
    return {}
