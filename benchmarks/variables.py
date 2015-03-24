# variables.py - Serialisable representation of experimental values.
#
# These classes represent experimental variables, and are designed for
# persistent storage through serialising/deserialising to and from
# JSON.
from time import time
from socket import gethostname

from util import checksum

#
class Variable:
    def encode(self):
        return {self.name: self.val}

    @staticmethod
    def decode(d):
        return IndependentVariable(d[0], d[1])

#
class IndependentVariable(Variable):
    def __init__(self, name, val):
        self.name = name
        self.val = val

    def __repr__(self):
        return "{name}: {val}".format(name=self.name, val=self.val)

#
class NullVariable(IndependentVariable):
    def __init__(self):
        IndependentVariable.__init__(self, "Null", None)

#
class Hostname(IndependentVariable):
    def __init__(self):
        val = gethostname()
        IndependentVariable.__init__(self, "Hostname", val)

#
class Argument(IndependentVariable):
    def __init__(self, val):
        IndependentVariable.__init__(self, "Argument", val)

    def __repr__(self):
        return str(self.val)

# Represents a tunable knob.
class Knob(IndependentVariable):
    #
    def build(self, val): pass

#
class DependentVariable(Variable):
    def __init__(self, name):
        self.name = name
        self.val = None

    # String representation of dependent variable.
    def __repr__(self):
        return "{name}: {val}".format(name=self.name, val=self.val)

    # Pre and post execution hooks.
    def pre(self, benchmark): pass
    def post(self, benchmark, exitstatus, output): pass

# A built-in runtime variable.
class RunTime(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Run time")

    def pre(self, benchmark):
        self.start = time()

    def post(self, benchmark, exitstatus, output):
        end = time()
        elapsed = end - self.start
        self.val = elapsed

# A built-in runtime variable.
class Checksum(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Checksum")

    def post(self, benchmark, exitstatus, output):
         self.val = checksum(benchmark.bin.path)

# A built-in runtime variable.
class ExitStatus(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Exit status")

    def post(self, benchmark, exitstatus, output):
        self.val = exitstatus

#
class LookupError(Exception):
    pass

#
def lookup(vars, type):
    return filter(lambda x: isinstance(x, type), vars)

#
def lookup1(*args):
    var = list(lookup(*args))
    if len(var) != 1:
        raise LookupError
    return var[0]
