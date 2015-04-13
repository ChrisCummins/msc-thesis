# resultscache.py - Persistent store for benchmark results.
#
# There are two public methods: load() and store().
#
# Results are stored in a three level hierarchy, where "key" is a hash
# of the set of independent variables for that result:
#
#     <benchmark>/<host>/<key>.json
from os.path import dirname
from util import path
from variables import BenchmarkName,Checksum,Hostname,lookup1,Result

from variables import HashableInvars

import config
import jsoncache

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
    return HashableInvars(invars).key()

#
def load(invars):
    path = resultspath(invars)
    data = jsoncache.load(path)

    return Result.decode(data, invars)

#
def store(result):
    path = resultspath(result.invars)
    jsoncache.store(path, result.encode())
