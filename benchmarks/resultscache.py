# resultscache.py - Persistent store for benchmark results.
#
# There are two public methods: load() and store().
#
# Results are stored in a three level hierarchy, where "key" is a hash
# of the set of independent variables for that result:
#
#     <benchmark>/<host>/<key>.json
from hashlib import sha1
from os.path import dirname
from util import path
from variables import BenchmarkName,Checksum,Hostname,lookup1,Result

import config
import jsoncache

#
class _HashableInvars:
    # A set of variable names to exclude from hash results.
    _EXCLUDED_KEYS = ["Hostname", "Benchmark"]

    def __init__(self, invars):

        # Filter out unhashable invars
        for key in self._EXCLUDED_KEYS:
            invars = filter(lambda x: not x.name == key, invars)

        self._invars = sorted(list(invars))
        self._key = sha1(str(self)).hexdigest()

    def key(self):
        return self._key

    def __key(x):
        return tuple(x._invars)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __repr__(x):
        return str(x.__key()).encode('utf-8')

#
def _path(benchmarkname, key, hostname, root, suffix, extension):
    return ("{root}/{benchmark}/{host}/{key}{suffix}{extension}"
            .format(root=root,
                    benchmark=benchmarkname,
                    host=hostname, key=key,
                    suffix=suffix, extension=extension))

#
def _loadresults(path):
    return jsoncache.load(path)

#
def _invars2path(invars, root=config.RESULTS, suffix="", extension=""):
    benchmark = lookup1(invars, BenchmarkName).val
    host = lookup1(invars, Hostname).val
    key = id(invars)
    return _path(benchmark, key, host, root, suffix, extension)

#
def resultspath(invars, suffix="", extension=".json"):
    return _invars2path(invars, root=config.RESULTS,
                        suffix=suffix, extension=extension)

#
def plotpath(invars, suffix="", extension=".svg"):
    return _invars2path(invars, root=config.PLOTS,
                        suffix=suffix, extension=extension)

#
def id(invars):
    return _HashableInvars(invars).key()

#
def load(invars):
    path = resultspath(invars)
    data = jsoncache.load(path)

    return Result.decode(data, invars)

#
def store(result):
    path = resultspath(result.invars)
    jsoncache.store(path, result.encode())
