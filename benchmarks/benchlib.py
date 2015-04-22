from __future__ import print_function
from collections import OrderedDict
from itertools import product
from math import sqrt
from os.path import basename,exists
from sys import stdout

import os

import config
from util import *
from variables import *
import resultscache

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

# Represents an OpenCL-capable host device. Accepts a list of
# platforms, each containing device descriptions, of the format:
#
# [
#   [
#     {
#       "name": <name>,
#       "version": <version>,
#       "opencl_version": <version>,
#       "compute_units": <uint>,
#       "clock_frequency": <uint>,
#       "global_memory_size": <uint>,
#       "local_memory_size": <uint>,
#       "max_work_group_size": <uint>,
#       "max_work_item_sizes": [<uint> ...],
#     },
#     ...
#   ]
#   ...
# ]
class OpenCLHost(Host):
    def __init__(self, name, platforms=[], **kwargs):
        Host.__init__(self, name, **kwargs)
        self.platforms = platforms

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

        cmd = [self.path]
        for arg in args:
            cmd += arg.val.split()
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
            Colours.print(Colours.RED, "FAIL")
            Colours.print(Colours.RED,
                          ("Process terminated with exit status {e}. Output:"
                           .format(e=exitstatus)))
            for l in output:
                print(l)

        # Return a tuple of dependent variables, constant dependent
        # variables, and a boolean of whether the program exited with
        # a nonzero exit status.
        return outvars, set(coutvars), exitstatus != 0

#
class SamplingPlan:
    def hasnext(self, result):
        return not result.bad

    def __repr__(self):
        return "Null"

# A straightforward sampling plan which continues sampling until
# "samplecount" has been reached.
class FixedSizeSampler(SamplingPlan):
    def __init__(self, samplecount=10):
        self.samplecount = samplecount

    def hasnext(self, result):
        # Check first with superclass.
        if SamplingPlan.hasnext(self, result):
            return len(result.outvars) < self.samplecount
        else:
            return False

    def __repr__(self):
        return "FixedSize({n})".format(n=self.samplecount)

# A variable length sampling plan which continues sampling until
# "minsamples" has been reached, then only if the normalised deviation
# is above "maxvariance", or if the number of samples reaches
# "maxsamples".
class MinimumVarianceSampler(SamplingPlan):
    def __init__(self, variable=RunTime, maxvariance=.15, minsamples=5, maxsamples=30):
        self.variable = variable
        self.maxvariance = maxvariance
        self.minsamples = minsamples
        self.maxsamples = maxsamples

    # Computer mean.
    def mean(self, l):
        return sum(l) / len(l)

    # Compute variance.
    def variance(self, l):
        if len(l) > 1:
            m = self.mean(l)
            return sum([(x - m) ** 2 for x in l]) / (len(l) - 1)
        else:
            return 0

    # Compute stdev.
    def stdev(self, l):
        return sqrt(self.variance(l))

    def hasnext(self, result):
        # Check first with superclass.
        if SamplingPlan.hasnext(self, result):
            numsamples = len(result.outvars)

            # Check that we have the minimum number of samples.
            if numsamples < self.minsamples: return True
            # Check that we haven't reached the maximum number of samples.
            if numsamples > self.maxsamples: return False

            # Get the values.
            vals = self.getvals(result)
            mean = self.mean(vals)
            stdev = self.stdev(vals)

            # Return wether the weighted deviation is great than acceptable.
            return stdev / mean > self.maxvariance
        else:
            return False

    # Return a list of values that we're interested in minimising the
    # variance of.
    def getvals(self, result):
        return [lookup1(x, self.variable).val for x in result.outvars]

    def __repr__(self):
        return "MinimumVarianceSampler({v}:{var})".format(var=self.variable,
                                                          v=self.maxvariance)


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
        self._hasresult = False # Don't load result until we use it
        self.host = lookup1(testcase.invars, Hostname).val # derive

    def run(self):
        # Only run if we are on the right host.
        if hostname() != self.host:
            return

        # Only run if we have samples to collect.
        if not self.sampler.hasnext(self.result()):
            return

        Colours.print(Colours.BLUE, "Preparing testcase", self.testcase, "...")
        self.testcase.setup()

        # Sample and store results.
        while self.sampler.hasnext(self._result):
            Colours.print(Colours.YELLOW, "Sampling testcase",
                          self.testcase, "... ", end="")
            stdout.flush()
            o, c, b = self.testcase.sample()
            print("{t:.1f} s".format(t=lookup1(o, RunTime).val)) # elapsed time
            self._result.outvars.append(o) # dep(endent) vars.
            self._result.couts.update(c) # constant dep vars.
            self._result.bad = b # "bad" flag.
            resultscache.store(self._result) # store new data.

        # Post-execution tidy-up.
        self.testcase.teardown()

    def result(self):
        if not self._hasresult:
            self._result = resultscache.load(self.testcase.invars)
            self._hasresult = True
        return self._result

    def __repr__(self):
        return ("{testcase}, Sampling plan: {sampler}"
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
    runnable = filter(lambda x: x.sampler.hasnext(x.result()), forthisdevice)

    print("From the {n} test cases, {t} are for this device, "
          "and {r} require samples."
          .format(n=len(harnesses), t=len(forthisdevice), r=len(runnable)))

    return list(runnable)

# Whether we've sent out the courtesy message.
_messagesent = False

def runJobQueue(harnesses):
    global _messagesent

    numjobs, i = len(harnesses), 1 # counters

    # Send out a courtesy message.
    if not _messagesent and hostname() not in config.MASTER_HOSTS:
        msg = "Beginning timed benchmarks (pid {pid})...".format(pid=pid())
        os.system('echo "{msg}" | wall'.format(msg=msg))
        _messagesent = True

    # Check we have something to do...
    if not len(harnesses): return

    Colours.print(Colours.GREEN, "Beginning execution of", numjobs,
                  "jobs ...")

    while harnesses:
        job = harnesses.pop()
        print()
        Colours.print(Colours.GREEN, "Beginning test harness", i, "of", numjobs)
        print()
        job.run()
        i += 1

# Separate a list of harnesses into collections who share the same
# values for the supplied list of independent variables.
def groupByInvars(harnesses, *args):
    grouped = OrderedDict()
    # Iterate over harnesses.
    for harness in harnesses:
        invars = harness.testcase.invars
        matched = [] # List of matching invars.
        for invar in args:
            try:
                matched.append(lookup1(invars, invar))
            except LookupError: # Ignore lookup errors
                pass
        # Create a key.
        key = HashableInvars(matched, exclude=[]).key()
        # Add key to dictionary.
        if key not in grouped:
            grouped[key] = []
        # Add harness to dictionary entry.
        grouped[key].append(harness)
    # Return the grouped harnesses.
    return grouped

def filterHarnessesByInvarVal(harnesses, type, val):
    output = []
    for harness in harnesses:
        try:
            harness_val = lookup1(harness.testcase.invars, type).val
            if val == harness_val:
                output.append(harness)
        except:
            pass
    return output
