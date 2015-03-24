# resultscache.py - Persistent store for benchmark results.
#
# There are two public methods: load() and store().
import jsoncache

from hashlib import sha1
from os.path import dirname
from util import path
from variables import BenchmarkName,Checksum,Hostname,lookup1,Result

# <benchmark>/<checksum>/<host>.json

ROOT = path(dirname(__file__) + "/results")

#
class _HashableInvars:
    def __init__(self, invars):
        self.invars = invars

    def key(self):
        return str(hash(self))

    def __key(x):
        return tuple(sorted(x.invars))

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(x):
        return int(sha1(str(x.__key()).encode('utf-8')).hexdigest(), 16)

# String
def _path(benchmarkname, key, hostname):
    return ("{root}/{benchmark}/{host}/{key}.json"
            .format(root=ROOT, benchmark=benchmarkname,
                    host=hostname, key=key))

#
def _loadresults(path):
    return jsoncache.load(path)

#
def load(testcase):
    benchmark = testcase.benchmark.name
    invars = testcase.invars
    host = lookup1(invars, Hostname).val
    key = _HashableInvars(invars).key()
    path = _path(benchmark, key, host)

    data = jsoncache.load(path)

    return Result.decode(data, invars)

#
def _resultpath(result):
    benchmark = lookup1(result.invars, BenchmarkName).val
    host = lookup1(result.invars, Hostname).val
    key = _HashableInvars(result.invars).key()
    return _path(benchmark, key, host)

#
def store(result):
    path = _resultpath(result)
    jsoncache.store(path, result.encode())
