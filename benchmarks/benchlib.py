from __future__ import print_function
from datetime import datetime
from hashlib import sha1
from itertools import product
from json import dump,dumps,load
from math import sqrt
from os import chdir,getcwd,listdir,makedirs
from os.path import abspath,dirname,exists
from random import shuffle
from re import match,search
from re import sub
from socket import gethostname
from subprocess import call,check_output
from sys import exit,stdout

##### LOCAL VARIABLES #####

# directory history
__cdhist = [dirname(__file__)]


##### UTILITIES #####

# Return the ID of the machine, used for identifying results.
def ID():
    return gethostname()

def skelcl_version():
    return check_output(['git', 'rev-parse', 'HEAD']).strip()

# Concatenate all components into a path.
def path(*components):
    return abspath('/'.join(components))

# Return the path to binary directory of example program "name".
def bindir(name):
    return path(SKELCL_BUILD, 'examples', name)

# Return the path to binary file of example program "name".
def bin(name):
    return path(bindir(name), name)

# Return the path of the results file for "version".
def resultsfile(version=skelcl_version()):
    return path(RESULTSDIR, '{0}.json'.format(version))

# Return the results in "file".
def loadresults(file):
    if exists(file):
        return load(open(file))
    else:
        return {}

# Store "results" in "file".
def store(results, file):
    dump(results, open(file, 'w'),
         sort_keys=True, indent=2, separators=(',', ': '))

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

# List all files and directories in "path". If "abspaths", return
# absolute paths.
def ls(p=".", abspaths=True):
    if abspath:
        return [abspath(path(p, x)) for x in listdir(p)]
    else:
        return listdir(p)

# Returns all of the lines in "file" as a list of strings, excluding
# comments (delimited by '#' symbol).
def parse(file):
    with open(file) as f:
        return [match('[^#]+', x).group(0).strip() for x in f.readlines() if not match('\s*#', x)]

# Return the current date in style "format".
def datestr(format="%I:%M%p on %B %d, %Y"):
    return datetime.now().strftime(format)

# Print the date and current working directory to "file".
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
DATADIR = path(CWD, 'data')
IMAGES = [x for x in  ls(path(DATADIR, 'img'))
          if search('\.pgm$', x) and not search('\.out\.pgm$', x)]
RESULTSDIR = path(CWD, 'results')
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

# Run program "prog" with arguments "args" and return the runtime.
#
#   @side-effect: Changes working dir.
def time(prog, args=[]):
    progdir, progbin = bindir(prog), bin(prog)

    cd(progdir)
    with open(RUNLOG, 'w') as f:
        printheader(f)
        system([progbin] + args, out=f)

    # Return execution time.
    for line in reversed(open(RUNLOG).readlines()):
        match = search('^Elapsed time:\s+([0-9]+)\s+', line)
        if match:
            return int(match.group(1))

# Record the runtime of "prog" using "args", under experiment
# "version".
def record(prog, args=[], version=skelcl_version()):
    f = resultsfile(version=version)
    R = loadresults(f)
    options = ' '.join(args)
    id = ID()

    if prog not in R:
        R[prog] = {}

    if options not in R[prog]:
        R[prog][options] = {}

    if id not in R[prog][options]:
        R[prog][options][id] = []

    R[prog][options][id].append(time(prog, args))
    store(R, f)

# Return all permutations of "options" for "prog"
def permutations(prog, options=[[]]):
    return list(product(*options))

# Run "prog" "n" times for all "options", where "options" is a list of
# lists, and "version" is the index for results.
def iterate(prog, options=[[]], n=30, version=skelcl_version()):
    P = permutations(prog, options)
    for i in range(n):
        for options in P:
            args = [item for sublist in [x.split() for x in options] for item in sublist]
            print(prog, *options)
            record(prog, args, version=version)
