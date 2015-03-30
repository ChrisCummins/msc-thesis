# variables.py - Serialisable representation of experimental values.
#
# These classes represent experimental variables, and are designed for
# persistent storage through serialising/deserialising to and from
# JSON.
from __future__ import print_function
from datetime import datetime
from time import time
from socket import gethostname
from inspect import isclass

from util import checksum

def _datefmt(datetime):
    _DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    return datetime.strftime(_DATETIME_FORMAT)

# A result consists of a set of independent variables, and one or more
# sets of dependent, or "output", variables.
class Result:
    def __init__(self, invars, outvars=[], couts=set(), bad=False):
        self.invars = invars
        self.outvars = outvars
        self.couts = couts
        self.bad = bad

    def __repr__(self):
        return '\n'.join([str(x) for x in self.invars + list(self.couts)] +
                         ['\n'.join(["    " + str(x) for x in y])
                          + '\n' for y in self.outvars])

    # Encode a result for JSON serialization.
    def encode(self):
        d = {'in': {}, 'cout': {}, 'out': [], 'bad': self.bad}

        # Create dictionaries.
        [d['in'].update(x.encode()) for x in self.invars]
        [d['cout'].update(x.encode()) for x in self.couts]

        # Build a list of derived variables.
        derived = filter(lambda x: isinstance(x, DerivedVariable), self.outvars[-1])
        d['dout'] = [x.encode() for x in derived]

        # Build a list of outvars dicts.
        for sample in self.outvars:
            # Filter out derived variables.
            encodable = filter(lambda x: not isinstance(x, DerivedVariable), sample)

            # Encode variables.
            o = {}
            [o.update(x.encode()) for x in encodable]
            d['out'].append(o)

        return d

    # Decode a serialesd JSON result.
    @staticmethod
    def decode(d, invars):
        outvars = [[DependentVariable(x, y[x]) for x in y] for y in d['out']] if 'out' in d else []
        couts = set([DependentVariable(x, d['cout'][x]) for x in d['cout']]) if 'cout' in d else set()
        bad = d['bad'] if 'bad' in d else False

        # FIXME: Pass module(s) to search.
        import skelcl
        doutvars = [getattr(skelcl, str(x)) for x in d['douts']] if 'douts' in d else []
        benchmark = lookup1(invars, "Benchmark").val
        doutcount = 0

        if len(doutvars):
            for sample in outvars:
                kwargs = {
                    'benchmark': lookup1(invars, "Benchmark").val,
                    'exitstatus': lookup1(sample, "Exit status").val,
                    'output': lookup1(sample, "Output").val
                }

                for doutvar in doutvars:
                    sample.append(doutvar(**kwargs))
                    doutcount += 1

        return Result(invars, outvars=outvars, couts=couts, bad=bad)


#
class Variable:
    def encode(self):
        return {self.name: self.val}

    def __repr__(self):
        return "{name}: {val}".format(name=self.name, val=self.val)

    def __key(x):
        return (x.name, x.val)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __lt__(x, y):
        if x.name == y.name:
            return x.val < y.val
        else:
            return x.name < y.name

    def __hash__(x):
        return hash(x.__key())

#
class IndependentVariable(Variable):
    def __init__(self, name, val):
        self.name = name
        self.val = val

#
class DependentVariable(Variable):
    def __init__(self, name, val=None):
        self.name = name
        self.val = val

    # String representation of dependent variable.
    def __repr__(self):
        return "{name}: {val}".format(name=self.name, val=self.val)

    # Pre and post execution hooks.
    def pre(self, **kwargs): pass
    def post(self, **kwargs): pass

#
class DerivedVariable(DependentVariable):
    def __init__(self, name, **kwargs):
        DependentVariable.__init__(self, name)

    def encode(self):
        return self.name

#########################
# Independent Variables #
#########################

#
class Hostname(IndependentVariable):
    def __init__(self, name):
        IndependentVariable.__init__(self, "Hostname", name)

#
class BenchmarkName(IndependentVariable):
    def __init__(self, name):
        IndependentVariable.__init__(self, "Benchmark", name)

# Runtime argument.
class Argument(IndependentVariable):
    pass

# Represents a tunable knob.
class Knob(IndependentVariable):
    # Set the value of the knob.
    def set(self): pass


#######################
# Dependent Variables #
#######################

# A built-in runtime variable.
class StartTime(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Start time")

    def pre(self, **kwargs):
        self.val = _datefmt(datetime.now())

# A built-in runtime variable.
class EndTime(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "End time")

    def post(self, **kwargs):
        self.val = _datefmt(datetime.now())

# A built-in runtime variable.
class RunTime(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Run time")

    def pre(self, **kwargs):
        self.start = time()

    def post(self, **kwargs):
        end = time()
        elapsed = end - self.start
        self.val = elapsed

# A built-in runtime variable.
class Checksum(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Checksum")

    def post(self, **kwargs):
         self.val = checksum(kwargs['benchmark'].bin.path)

# A built-in runtime variable.
class ExitStatus(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Exit status")

    def post(self, **kwargs):
        self.val = kwargs['exitstatus']

# A built-in runtime variable.
class Output(DependentVariable):
    def __init__(self):
        DependentVariable.__init__(self, "Output")

    def post(self, **kwargs):
        self.val = kwargs['output']


###########
# Filters #
###########

#
class LookupError(Exception):
    pass

#
def lookup(vars, type):
    if isclass(type):
        f = filter(lambda x: isinstance(x, type), vars)
    else:
        f = filter(lambda x: x.name == type, vars)
    # Evaluate filter into a list of results:
    return list(f)

#
def lookup1(*args):
    var = list(lookup(*args))
    if len(var) != 1:
        raise LookupError
    return var[0]
