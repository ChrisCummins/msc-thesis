from __future__ import print_function
from atexit import register
from datetime import datetime
from itertools import product
from os import chdir,getcwd,listdir,makedirs
from os.path import abspath,basename,dirname,exists
from random import shuffle
from re import match,search
from re import sub
from subprocess import call,check_output
from sys import exit,stdout

import json
import os

import config
from util import *
from variables import *
from stats import *
import jsoncache
import resultscache


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

####### STATS #######



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
        self._specs = [
            "{cpu}".format(cpu=cpu),
            "{g} GiB memory".format(g=mem)
        ]
        if len(gpus): # List GPUs if available.
            self._specs.append("GPUs: {{{0}}}".format(", ".join(gpus)))

    def __repr__(self):
        return "{host}: {specs}".format(host=self.NAME,
                                        specs=", ".join(self._specs))

# Represents an OpenCL-capable host device. All GPUs are assumed to
# have OpenCL drivers. "opencl_cpu" sets whether the CPU is OpenCL
# accelerated.
class OpenCLHost(Host):
    def __init__(self, name, opencl_cpu=False, **kwargs):
        Host.__init__(self, name, **kwargs)
        self.OPENCL_CPU = opencl_cpu
        if opencl_cpu:
            self._specs[0] += " w/ OpenCL"

# Represents the current host machine.
class LocalHost(Host):
    def __init__(self, **kwargs):
        Host.__init__(self, hostname(), **kwargs)

# Represents an application binary, at path "path".
class Binary:
    def __init__(self, path):
        self.basename = basename(path)
        self.path = path
        self._haschecksum = False
        self._checksum = ""

    def checksum(self):
        if not self._haschecksum:
            self._checksum = checksum(self.path)
            self._haschecksum = True

        return self._checksum

    def run(self, args):
        if not exists(self.path):
            raise Exception("Binary '{path}' does not exist!"
                            .format(path=self.path))

        cmd = [self.path] + [str(arg.val) for arg in args]
        return system(cmd, out=open(config.RUNLOG, 'w'), exit_on_error=False)

    def __repr__(self):
        return self.basename

#
class Benchmark:
    def __init__(self, name, path):
        self.name = name
        self.bin = Binary(path)

    def __repr__(self):
        return str(self.name)

    def build(self):
        pass

    # Run the benchmark and set "outvars".
    def run(self, args, outvars, coutvars):
        # Instantiate variables.
        coutvars = [var() for var in coutvars]
        outvars = [var() for var in outvars]

        # Pre-exection hooks.
        kwargs = {
            'benchmark': self
        }
        [var.pre(**kwargs) for var in outvars]
        [var.pre(**kwargs) for var in coutvars]

        exitstatus = self.bin.run(args)
        output = [l.rstrip() for l in open(config.RUNLOG).readlines()]

        # Post-execution hooks.
        kwargs = {
            'benchmark': self,
            'exitstatus': exitstatus,
            'output': output
        }
        [var.post(**kwargs) for var in outvars]
        [var.post(**kwargs) for var in coutvars]

        if exitstatus:
            Colours.print(Colours.RED,
                          ("Process terminated with exit status {e}. Output:"
                           .format(e=exitstatus)))
            for l in output:
                print(l)

        return outvars, set(coutvars)

#
class Sampler:
    def hasnext(result): return True

    def __repr__(self):
        return "Null"

#
class FixedSizeSampler(Sampler):
    def __init__(self, samplecount=10):
        self.samplecount = samplecount

    def hasnext(self, result):
        return len(result.outvars) < self.samplecount

    def __repr__(self):
        return "FixedSize({n})".format(n=self.samplecount)

#
class TestCase:
    def __init__(self, benchmark, host=LocalHost(),
                 invars=[], outvars=[], coutvars=set()):
        # Default variables.
        ins = [ # independent
            Hostname(host.NAME),
            BenchmarkName(benchmark.name)
        ]
        outs = [
            StartTime,
            EndTime,
            RunTime,
            ExitStatus,
            Output
        ]
        couts = {
            Checksum
        }

        self.benchmark = benchmark
        self.invars = ins + invars
        self.outvars = outs + outvars
        self.coutvars = couts.union(coutvars)

        # Arguments cache.
        self._hasargs = False
        self._args = []

        # Knobs cache.
        self._hasknobs = False
        self._knobs = []

    def setup(self):
        # Get knobs.
        if not self._hasknobs:
            self._knobs = lookup(self.invars, Knob)
            self._hasknobs = True

        # Set the value of tunable knobs.
        for knob in self._knobs:
            print("Setting", knob, "...")
            knob.set()

        # Build the benchmark.
        print("Building", self.benchmark.name, "...")
        self.benchmark.build()

    def teardown(self):
        pass

    def sample(self):
        # Get arguments.
        if not self._hasargs:
            self._args = lookup(self.invars, Argument)
            self._hasargs = True

        return self.benchmark.run(self._args, self.outvars, self.coutvars)

    def __repr__(self):
        return ", ".join([str(x) for x in self.invars])

#
class TestHarness:
    def __init__(self, testcase, sampler=FixedSizeSampler()):
        self.testcase = testcase
        self.sampler = sampler
        self.result = resultscache.load(testcase)
        self.host = lookup1(testcase.invars, Hostname).val # derive

    def run(self):
        # Only run if we are on the right host.
        if hostname() != self.host:
            return

        # Only run if we have samples to collect.
        if not self.sampler.hasnext(self.result):
            return

        Colours.print(Colours.BLUE, "Preparing testcase", self.testcase, "...")
        self.testcase.setup()

        # Sample and store results.
        while self.sampler.hasnext(self.result):
            Colours.print(Colours.YELLOW, "Sampling testcase",
                          self.testcase, "...")
            o, c = self.testcase.sample()
            self.result.outvars.append(o)
            self.result.couts.update(c)
            resultscache.store(self.result)

        # Post-execution tidy-up.
        self.testcase.teardown()

    def __repr__(self):
        return ("{testcase}, Sampler: {sampler}"
                .format(sampler=self.sampler, testcase=self.testcase))

def variablerange(Variable, vals, *args, **kwargs):
    return [Variable(x, *args, **kwargs) for x in vals]

def permutations(*args):
    return list(product(*args))

class TestGenerator:
    pass

def jobqueue(harnesses):
    localhost = hostname()
    forthisdevice = filter(lambda x: x.host == localhost, harnesses)
    runnable = filter(lambda x: x.sampler.hasnext(x.result), forthisdevice)
    return list(runnable)

def runJobQueue(harnesses):
    numjobs, i = len(harnesses), 1 # counters

    Colours.print(Colours.GREEN, "Beginning execution of", numjobs,
                  "jobs ...")

    while harnesses:
        job = harnesses.pop()
        print()
        Colours.print(Colours.GREEN, "Beginning test harness", i, "of", numjobs)
        print()
        job.run()
        i += 1
