# resultscache.py - Persistent store for benchmark results.
#
# There are two public methods: load() and store().
import jsoncache

from os.path import dirname
from util import path
from variables import Checksum,Hostname,lookup1

# <benchmark>/<checksum>/<host>.json

ROOT = path(dirname(__file__) + "/results")

# String
def _path(benchmark, checksum, host):
    return ("{root}/{benchmark}/{checksum}/{host}.json"
            .format(root=ROOT, benchmark=benchmark,
                    checksum=checksum, host=host))

#
def _loadresults(path):
    data = jsoncache.load(path)
    if 'results' not in data:
        data['results'] = []
    return data['results']

#
def load(benchmark, checksum, host):
    path = _path(benchmark, checksum, host)
    return _loadresults(path)

#
def _resultpath(result):
    benchmark = result.benchmark.name
    checksum = lookup1(result.outvars, Checksum).val
    host = lookup1(result.invars, Hostname).val

    return _path(benchmark, checksum, host)

#
def store(result):
    path = _resultpath(result)
    results = _loadresults(path)
    results.append(result.encode())
    jsoncache.store(path)
