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
from skelcl import *

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
    skeltimes = {"submit": [], "run": [], "queue": []}
    conttimes = {"ul": {"submit": [], "run": [], "queue": []},
                 "dl": {"submit": [], "run": [], "queue": []}}

    def parsesample(sample):
        st = lookup(sample, SkeletonEventTimes)
        ct = lookup(sample, ContainerEventTimes)

        for var in st:
            val = var.val
            for type in val:
                for address in val[type]:
                    for event in val[type][address]:
                        skeltimes["queue"].append(event[0])
                        skeltimes["submit"].append(event[1])
                        skeltimes["run"].append(event[2])

        for var in ct:
            val = var.val
            for type in val:
                for address in val[type]:
                    for direction in val[type][address]:
                        for event in val[type][address][direction]:
                            conttimes[direction]["queue"].append(val[type][address][direction][event][0])
                            conttimes[direction]["submit"].append(val[type][address][direction][event][1])
                            conttimes[direction]["run"].append(val[type][address][direction][event][2])

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
    data = [
        ("UL-queue", describe(conttimes["ul"]["queue"])),
        ("UL-submit", describe(conttimes["ul"]["submit"])),
        ("UL-run", describe(conttimes["ul"]["run"])),
        ("Skel-queue", describe(skeltimes["queue"])),
        ("Skel-submit", describe(skeltimes["submit"])),
        ("Skel-run", describe(skeltimes["run"])),
        ("DL-queue", describe(conttimes["dl"]["queue"])),
        ("DL-submit", describe(conttimes["dl"]["submit"])),
        ("DL-run", describe(conttimes["dl"]["run"]))
    ]

    Y, Yerr, Labels = zip(*[(x[1][0], x[1][1], x[0]) for x in data])
    X = np.arange(len(Y))

    width = 1

    ax = plt.axes()

    # Plot the data. Note the positive zorder.
    plt.bar(X, Y, width, yerr=Yerr,
            color=['red', 'yellow', 'green'], ecolor='k')
    ax.yaxis.grid(b=True, which='major', color="#aaaaaa", linestyle='-')

    # Set the graph bounds.
    plt.gca().set_position((.08, # Left padding
                            .26, # Bottom padding
                            .9, # Width
                            .68)) # Height

    # Set the caption text.
    plt.figtext(.02, .02, ("Total: {total} ms.".format(total=sum(Y))))

    plt.xlabel('Event type')
    plt.ylabel('Time (ms)')
    plt.title('OpenCL events: {v}'.format(v=', '.join([str(x.val) for x in invars])),
              fontsize=12, weight="bold")
    plt.xticks(X + width / 2., Labels, rotation=90)

    _finalize(result, name)
