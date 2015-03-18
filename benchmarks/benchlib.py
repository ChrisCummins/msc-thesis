from __future__ import print_function
from atexit import register
from datetime import datetime
from hashlib import sha1
from itertools import product
from math import sqrt
from os import chdir,getcwd,listdir,makedirs
from os.path import abspath,dirname,exists
from random import shuffle
from re import match,search
from re import sub
from socket import gethostname
from subprocess import call,check_output
from sys import exit,stdout

import scipy
from scipy import stats

import json

##### LOCAL VARIABLES #####

# directory history
__cdhist = [dirname(__file__)]

# Experimental results are cached in an dictionary, indexed by the
# experiment name. Results are read and written using the load() and
# store() functions, respectively. On a cache miss, the load()
# function looks up the name of the results file for that experiment
# and, if found, adds it to the cache. A store() marks that experiment
# as "cachedirty", and increments the "cachewrites" counter by 1. When
# the value of "cachewrites" reaches "cachewritethreshold", the cache
# is emptied and any file in the "cachedirty" set is written.
_cache = {}
_cachedirty = set()
_cachewrites = 0
_cachewritethreshold = 5

##### UTILITIES #####

# Return the ID of the machine, used for identifying results.
def ID():
    return gethostname()

# Get the current SkelCL git version.
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

def pprint(data):
    return json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '))

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
        return [match('[^#]+', x).group(0).strip()
                for x in f.readlines() if not match('\s*#', x)]

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


# Returns the hex checksum of "file".
def checksum(file):
    return sha1(open(file).read()).hexdigest()


####### STATS #######

# Return the mean value of a list of divisible numbers.
def mean(num):
    if len(num):
        return sum([float(x) for x in num]) / float(len(num))
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

# Return the confidence interval of a list for a given confidence
def confinterval(l, c=0.95, n=30):
    if len(l) > 1:
        scale = stdev(l) / sqrt(len(l))

        if len(l) >= n:
            # For large values of n, use a normal (Gaussian) distribution:
            c1, c2 = scipy.stats.norm.interval(c, loc=mean(l), scale=scale)
        else:
            # For small values of n, use a t-distribution:
            c1, c2 = scipy.stats.t.interval(c, len(l) - 1, loc=mean(l), scale=scale)

        return c1, c2
    else:
        return 0, 0


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


###### EXPERIMENTAL RESULTS #####

# Return the path of the results file for "version".
def _versionfile(version=skelcl_version()):
    return path(RESULTSDIR, '{0}.json'.format(version))

# Write cached data for "version" to disk.
def _writeversion(version=skelcl_version()):
    file = _versionfile(version)
    json.dump(_cache[version], open(file, 'w'),
              sort_keys=True, indent=2, separators=(',', ': '))
    print("Wrote '{0}'...".format(file))

# Load data for "version" to cache.
def _loadcache(version=skelcl_version()):
    file = _versionfile(version)
    if exists(file):
        data = json.load(open(file))
        print("Read '{0}'...".format(file))
    else:
        data = {}

    _cache[version] = data


# Dump dirty cached data to disk.
def _dumpdirty():
    global _cachedirty, _cachewrites

    for version in _cachedirty:
        _writeversion(version)

    _cachewrites = 0
    _cachedirty = set()
    print("Results cache clean.")

# Dump results cache.
def _dumpcache():
    if not _cachewrites: # there may be nothing to dump
        return
    _dumpdirty()

# Mark "version" as cache dirty.
def _flagdirty(version):
    global _cachewrites, _cachedirty
    _cachedirty.add(version)
    _cachewrites += 1

# Register exit handler.
register(_dumpcache)

# Return the results for "version".
def load(version):
    if version not in _cache:
        _loadcache(version)
    return _cache[version]

# Store "results" for "version".
def store(results, version):
    _flagdirty(version)
    # If there's enough dirty data, dump cache.
    if _cachewrites >= _cachewritethreshold:
        _dumpdirty()

###### BENCHMARK FUNCTIONS #####

def buildlog():
    return open(BUILDLOG, 'w')

# Build example program "prog", an log output to "log". If "clean",
# then clean before building.
#
#   @side-effect: Changes working dir.
def make(prog, clean=True, log=buildlog()):
    cd(bindir(prog))
    printheader(log)
    if clean:
        system(['make', 'clean'], out=log)
    system(['make', prog], out=log)

# Build SkelCL. If "configure", run cmake.
#
#   @side-effect: Changes working dir.
def makeSkelCL(configure=True, clean=True, log=buildlog()):
    cd(SKELCL_BUILD)
    printheader(log)
    if configure:
        system(['cmake', '..'], out=log)
    if clean:
        system(['make', 'clean'], out=log)
    system(['make'], out=log)

# Run "prog" with "args", and log output to "log". Returns exit status
# of program.
#
#   @side-effect: Changes working dir.
def runprog(prog, args, log=open(RUNLOG, 'w')):
    cd(bindir(prog))
    printheader(log)
    return system([bin(prog)] + args, out=log, exit_on_error=False)

# Parse a program output and return a list of runtimes (in ms).
def parseruntimes(output):
    r = []
    for line in output:
        match = search('^Elapsed time:\s+([0-9]+)\s+', line)
        if match:
            r.append(int(match.group(1)))
    return r

# Run program "prog" with arguments "args" and an error of
# runtime(s). If prog produces no runtimes, return value "ebad".
#
#   @side-effect: Changes working dir.
def times(prog, args=[], ebad=[-1]):
    e = runprog(prog, args, log=open(RUNLOG, 'w'))
    if e:
        print("Died, with output:")
        [print(x.rstrip()) for x in open(RUNLOG).readlines()]
        return ebad

    r = parseruntimes(open(RUNLOG).readlines())
    return r if len(r) else ebad

# Lookup results for "prog" with "args" (where "args" is either a
# concatenated string or a list) on "id" under "version".
def results(prog, args, id=ID(), version=skelcl_version()):
    R = load(version)
    if not isinstance(args, basestring):
        args = ' '.join(args)

    if prog in R:
        if args in R[prog]:
            if id in R[prog][args]:
                return R[prog][args][id]
    return []

# Record the runtimes of "prog" using "args", under experiment
# "version".
def record(prog, args=[], version=skelcl_version(), n=-1, count=-1):
    R = load(version)
    options = ' '.join(args)
    id = ID()

    # If we don't have enough results
    if n < 0 or len(results(prog, args, version=version)) < n:
        # Print out header.
        if count >= 0:
            print("iteration", count, end=" ")
        print("[{0}/{1}] {2} {3}"
              .format(len(results(prog, args, version=version)), n,
                      prog, options))

        t = times(prog, args)
        if t[0] >= 0:
            if prog not in R:
                R[prog] = {}

            if options not in R[prog]:
                R[prog][options] = {}

            if id not in R[prog][options]:
                R[prog][options][id] = []

            R[prog][options][id] += t
            store(R, version)

# Return all permutations of "options" for "prog"
def permutations(options=[[]]):
    return [[x for z in [x.split() for x in y] for x in z]
            for y in list(product(*options))]

# Run "prog" "n" times for all "options", where "options" is a list of
# lists, and "version" is the index for results.
def iterate(experiment):
    progs = experiment['progs']
    name = experiment['name']
    n = experiment['iterations']

    permcount = sum([len(permutations(progs[prog])) for prog in progs])
    itercount = permcount * n
    count = 0

    print("========================")
    print("SETTINGS:")
    for s in experiment['settings-val']:
        print("    {0}: {1}".format(s, experiment['settings-val'][s]))
    print()
    print("TOTAL PERMUTATIONS:", permcount)
    print("TOTAL ITERATIONS:  ", itercount)
    print("========================\n")

    for i in range(1, n + 1):
        for prog in progs:
            P = permutations(progs[prog])

            for args in P:
                count += 1
                record(prog, args, version=name, n=n, count=count)


def settingspermutations(options):
    vals = [x for x in options]
    keys = product(*[options[x] for x in options])
    return [dict(zip(vals, k)) for k in keys]

def runexperiment(experiment):
    basename = experiment['name']

    for p in settingspermutations(experiment['settings']):
        experiment['pre-exec-hook'](p)
        experiment['name'] = "{0}-{1}".format(basename, "-".join([str(p[x]) for x in p]))
        experiment['settings-val'] = p
        iterate(experiment)

def json2arff(schema, data, relation="data", file=stdout):
    print("@RELATION {0}".format(relation), file=file)
    print(file=file)

    for attribute in schema:
        print("@ATTRIBUTE {0} {1}"
              .format(attribute[0], attribute[1]), file=file)

    print(file=file)
    print("@DATA", file=file)
    for d in data:
        dd = [str(x) for x in d]
        print(','.join(dd), file=file)

    if file != stdout:
        print("Wrote '{0}'...".format(file.name))
