from __future__ import print_function
from atexit import register
from datetime import datetime
from itertools import product
from math import sqrt
from os import chdir,getcwd,listdir,makedirs
from os.path import abspath,basename,dirname,exists
from random import shuffle
from re import match,search
from re import sub
from scipy import stats
from subprocess import call,check_output
from sys import exit,stdout

import json
import scipy

import jsoncache
import resultscache
from util import checksum
from variables import *

##### LOCAL VARIABLES #####

# directory history
__cdhist = [dirname(__file__)]

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

def mkdir(path):
    try:
        makedirs(path)
    except OSError:
        pass

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


##### OBJECT ORIENTATION #####

# Represents a host device, i.e. a machine to collect results
# from. Host objects should be immutable.
class Host:
    def __init__(self, name, cpu="", mem="", gpus=[]):
        self.NAME = name
        self.CPU = cpu
        self.MEM = mem
        self.GPUS = gpus

        # Compute the string.
        _mem = "{g} GiB".format(g=mem)
        _gpus = "{{{0}}}".format(", ".join(gpus)) if len(gpus) else ""
        self._str = ("{host}: {hw}"
                     .format(host=name, hw=", ".join([cpu, _mem, _gpus])))

    def __repr__(self):
        return self._str

#
class Binary:
    def __init__(self, path, runlog):
        if not exists(path):
            raise Exception("Binary '{path}' does not exist!"
                            .format(path=path))

        self.basename = basename(path)
        self.path = path
        self.runlog = runlog
        self._haschecksum = False
        self._checksum = ""

    def checksum(self):
        if not self._haschecksum:
            self._checksum = checksum(self.path)
            self._haschecksum = True

        return self._checksum

    def run(self, args):
        cmd = [self.path] + [str(arg) for arg in args]
        return system(cmd, out=open(self.runlog, 'w'), exit_on_error=False)

    def __repr__(self):
        return self.basename

#
class Benchmark:
    def __init__(self, name, path, logfile, outvars=[]):
        self.name = name
        self.logfile = logfile
        self.bin = Binary(path, logfile)
        self._builtins = [RunTime(), Checksum(), ExitStatus()]
        self.outvars = outvars

    def __repr__(self):
        return str(self.name)

    def run(self, args):
        outvars = self._builtins + self.outvars

        [var.pre(self) for var in outvars]
        exitstatus = self.bin.run(args)
        output = [l.rstrip() for l in open(self.logfile).readlines()]
        [var.post(self, exitstatus, output) for var in outvars]

        return outvars

#
class Result:
    def __init__(self, benchmark, invars, outvars):
        self.benchmark = benchmark
        self.invars = invars
        self.outvars = outvars

    def __repr__(self):
        return " ".join([str(x) for x in
                         [self.benchmark, self.invars, self.outvars]])

    # Encode a result for JSON serialization.
    def encode(self):
        d = {}
        [d.update(x.encode()) for x in self.invars + self.outvars]
        return d

#
class Sampler:
    def hasnext(results): return True

#
class FixedSizeSampler(Sampler):
    def __init__(self, samplecount=10):
        self.samplecount = samplecount

    def hasnext(self, results):
        n = len(results)
        run = len(results) < self.samplecount
        if run:
            print("Has {n}/{total} results. Running.".
                  format(n=n, total=self.samplecount))
        return run

#
class TestCase:
    def __init__(self, benchmark, invars=[]):
        # Built in independent variables.
        builtins = [Hostname()]

        self.benchmark = benchmark
        self.invars = builtins + invars

        # Arguments cache.
        self._hasargs = False
        self._args = []

    def sample(self):
        args = (self._args if self._hasargs else
                filter(lambda x: isinstance(x, Argument), self.invars))
        self._hasargs = True # Mark arguments as loaded
        outvars = self.benchmark.run(args)
        return Result(self.benchmark, self.invars, outvars)

#
class TestHarness:
    def __init__(self, host, testcase, sampler=FixedSizeSampler()):
        self.host = host
        self.testcase = testcase
        self.sampler = sampler

    def run(self):
        # Only run if we are on the right host.
        if gethostname() != self.host.NAME:
            return

        print("Running", self, "...")
        while self.sampler.hasnext(self.results()):
            result = self.testcase.sample()
            resultscache.store(result)

    def results(self):
        return resultscache.load(self.testcase.benchmark,
                                 self.testcase.benchmark.bin.checksum(),
                                 self.host.NAME)

#
class TestSuite:
    def __init__(hosts, harnesses):
        self.hosts = hosts
        self.harnesses = harnesses

##### SKELCL-SPECIFIC CLASSES #####

#
class SkelCLElapsedTimes(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Elapsed times")

    def post(self, benchmark, exitstatus, output):
        r = []
        for line in output:
            match = search('^Elapsed time:\s+([0-9]+)\s+', line)
            if match:
                r.append(int(match.group(1)))

        # Clamp each time at >= 1 ms. This is necessary for computations
        # that require less than .5 ms, which will be rounded down to 0.
        self.val = [max(x, 1) for x in r]

#
class SkelCLBenchmark(Benchmark):
    def __init__(self, name, outvars=[]):
        # Path to directory:
        self.dir = path(SKELCL_BUILD, 'examples', name)
        # Path to binary:
        binpath = path(self.dir, name)

        # Builtin dependent variables.
        outvars = [SkelCLElapsedTimes()] + outvars

        # Superclass:
        Benchmark.__init__(self, name, binpath, RUNLOG,
                           outvars=outvars)

    def run(self, args):
        cd(self.dir)
        return Benchmark.run(self, args)
