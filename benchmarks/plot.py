from __future__ import print_function
from hashlib import sha1
import matplotlib.pyplot as plt
import numpy as np

import resultscache
from variables import lookup,lookup1
from os.path import dirname,exists
from re import compile,match

from util import Colours,mkdir
from stats import *

#
class _HashableResult:
    def __init__(self, result):
        self._vars = sorted(list(result.outvars))
        self._key = sha1(str(self)).hexdigest()

    def key(self):
        return self._key

    def __key(x):
        return tuple(x._vars)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __repr__(x):
        return str(x.__key()).encode('utf-8')

#
def _gettimes(samples):
    skeltimes = {"submitTime": [], "runTime": [], "queueTime": []}
    conttimes = {"upload": {"submitTime": [], "runTime": [], "queueTime": []},
                 "download": {"submitTime": [], "runTime": [], "queueTime": []}}

    def parsesample(sample):
        st = lookup(sample, "Skeleton Event timings")
        ct = lookup(sample, "Container Event timings")

        for t in st:
            for skel in t.val:
                for event in t.val[skel]['events']:
                    skeltimes["submitTime"].append(event['submitTime'])
                    skeltimes["runTime"].append(event['runTime'])
                    skeltimes["queueTime"].append(event['queueTime'])

        for t in ct:
            for cont in t.val:
                for addr in t.val[cont]:
                    for direction in t.val[cont][addr]:
                        for event in t.val[cont][addr][direction]:
                            for time in t.val[cont][addr][direction][event]:
                                conttimes[direction][time].append(t.val[cont][addr][direction][event][time])

    [parsesample(x) for x in samples]
    return skeltimes, conttimes

def _writechecksum(path, checksum):
    file = open(path, 'a')
    file.write("<!-- __checksum__ {checksum} -->"
               .format(checksum=checksum))
    file.close()

_checksumre = compile("<!-- __checksum__ ([0-9a-f]+) -->")

def _readchecksum(path):
    # The checksum is embedded in the last line of image data, so we
    # employ a speedy hack to read the last line using seek().
    #
    # See: http://stackoverflow.com/a/3346492/1318051
    with open(path, 'rb') as file:
        first = next(file).decode()

        file.seek(-1024, 2)
        last = file.readlines()[-1].decode()
        return _checksumre.match(last).group(1)

#
def _finalize(result, name):
    path = resultscache.plotpath(result.invars, suffix="-{name}".format(name=name))
    mkdir(dirname(path))
    Colours.print(Colours.BLUE, "Wrote {path} ...".format(path=path))
    plt.savefig(path)
    plt.close()
    # Embed checksum of graphed data in the plot file.
    _writechecksum(path, _HashableResult(result).key())

#
def _skippable(result, name):
    if result.bad:
        Colours.print(Colours.RED, "warning: skipping plot of bad data")
        return True
    if not len(result.outvars):
        Colours.print(Colours.RED, "warning: skipping plot because there's no data")
        return True

    # Check that graphed data has been modified.
    path = resultscache.plotpath(result.invars, suffix="-{name}".format(name=name))
    if exists(path):
        if _readchecksum(path) == _HashableResult(result).key():
            return True

    return False

#
def openCLEventTimes(invars, name="events"):
    result = resultscache.load(invars)
    if _skippable(result, name): return

    skeltimes, conttimes = _gettimes(result.outvars)
    times = {
        "skel-queue": describe(skeltimes["queueTime"]),
        "skel-submit": describe(skeltimes["submitTime"]),
        "skel-run": describe(skeltimes["runTime"]),
        "dl-queue": describe(conttimes["download"]["queueTime"]),
        "dl-submit": describe(conttimes["download"]["submitTime"]),
        "dl-run": describe(conttimes["download"]["runTime"]),
        "ul-queue": describe(conttimes["upload"]["queueTime"]),
        "ul-submit": describe(conttimes["upload"]["submitTime"]),
        "ul-run": describe(conttimes["upload"]["runTime"]),
    }

    Y, Yerr, Labels = zip(*[(times[x][0], times[x][1], x) for x in sorted(times)])
    X = np.arange(len(Y))

    width = 0.35
    p1 = plt.bar(X, Y, width, yerr=Yerr, ecolor='k')

    plt.xlabel('Event type')
    plt.ylabel('Time (ms)')
    plt.title('OpenCL event timings: {v}'.format(v=', '.join([str(x.val) for x in invars])))
    plt.xticks(X + width / 2., Labels)
    #plt.yticks(np.arange(0,81,10))
    #plt.legend( (p1[0], p2[0]), ('Men', 'Women') )

    _finalize(result, name)
