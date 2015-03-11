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

import json

##### LOCAL VARIABLES #####

# directory history
__cdhist = [dirname(__file__)]

# Experimental results are stored in-memory in a cache, indexed by the
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


###### EXPERIMENTAL RESULTS #####

# Return the path of the results file for "version".
def _versionfile(version=skelcl_version()):
    return path(RESULTSDIR, '{0}.json'.format(version))

# Dump results cache.
def _dumpcache_():
    global _cachedirty, _cachewrites

    if not _cachewrites: # there may be nothing to dump
        return

    for version in _cachedirty:
        file = _versionfile(version)
        json.dump(_cache[version], open(file, 'w'),
                  sort_keys=True, indent=2, separators=(',', ': '))
        print("Wrote '{0}'...".format(file))

    _cachewrites = 0
    _cachedirty = set()
    print("Results cache clean.")

# Register and exit handler.
register(_dumpcache_)

# Return the results for "file".
def load(version):
    if version in _cache:
        return _cache[version]
    else:
        file = _versionfile(version)
        if exists(file):
            _cache[version] = json.load(open(file))
        else:
            _cache[version] = {}
        return load(version) # recurse

# Store "results" for "version".
def store(results, version):
    global _cachewrites
    _cachedirty.add(version)
    _cachewrites += 1
    if _cachewrites >= _cachewritethreshold:
        _dumpcache_()

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

# Run program "prog" with arguments "args" and an error of runtime(s).
#
#   @side-effect: Changes working dir.
def times(prog, args=[]):
    progdir, progbin = bindir(prog), bin(prog)

    cd(progdir)
    with open(RUNLOG, 'w') as f:
        printheader(f)
        e = system([progbin] + args, out=f, exit_on_error=False)
    if e:
        print("Died.")
        return -1

    # Return execution times.
    r = []
    for line in reversed(open(RUNLOG).readlines()):
        match = search('^Elapsed time:\s+([0-9]+)\s+', line)
        if match:
            r.append(int(match.group(1)))

    if len(r):
        return r
    else:
        return [-1]

# Lookup results for "prog" with "args" on "id" under "version".
def results(prog, args, id=ID(), version=skelcl_version()):
    R = load(version)
    options = ' '.join(args)

    if prog in R:
        if options in R[prog]:
            if id in R[prog][options]:
                return R[prog][options][id]
    return []

# Record the runtimes of "prog" using "args", under experiment
# "version".
def record(prog, args=[], version=skelcl_version(), n=-1, count=-1):
    R = load(version)
    options = ' '.join(args)
    id = ID()

    # If we don't have enough results
    if n < 0 or len(results(prog, args, version=version)) <= n:
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
    print("TOTAL PERMUTATIONS:", permcount)
    print("TOTAL ITERATIONS:  ", itercount)
    print("========================\n")

    for i in range(1, n + 1):
        for prog in progs:
            P = permutations(progs[prog])

            for args in P:
                count += 1
                record(prog, args, version=name, n=n, count=count)


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
