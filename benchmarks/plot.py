from __future__ import print_function
from hashlib import sha1
from textwrap import wrap
from matplotlib import rc

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
    inittimes = []
    buildtimes = []
    preptimes = []
    swaptimes = []
    skeltimes = []
    conttimes = {"ul": [], "dl": []}

    def parsesample(sample):
        it = lookup1(sample, InitTime)
        bt = lookup1(sample, ProgramBuildTimes)
        pt = lookup1(sample, PrepareTimes)
        swt = lookup1(sample, SwapTimes)
        st = lookup(sample, SkeletonEventTimes)
        ct = lookup(sample, ContainerEventTimes)
        ndevices = len(lookup1(sample, Devices).val)

        inittimes.append(it.val)
        buildtimes.append(sum(bt.val))
        swaptimes.append(sum(swt.val))

        for type in pt.val:
            for address in pt.val[type]:
                preptimes.append(sum(pt.val[type][address]))

        # Collect skeleton and container OpenCL event times. Note here
        # that we are first summing up the total times for *all*
        # events of each type, and that each event time is divided by
        # the number of devices.

        # Skeleton times
        for var in st:
            val = var.val
            for type in val:
                for address in val[type]:
                    skeltimes.append(sum(val[type][address]) / ndevices)

        # Container upload and download times.
        for var in ct:
            val = var.val
            for type in val:
                for address in val[type]:
                    for direction in val[type][address]:
                        times = [val[type][address][direction][x]
                                 for x in val[type][address][direction]]
                        conttimes[direction].append(sum(times) / ndevices)

    [parsesample(x) for x in samples]
    return inittimes, buildtimes, preptimes, swaptimes, skeltimes, conttimes

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
        match = _checksumre.match(last)
        return match.group(1) if match else None

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
        # Bad data is worth warning about.
        Colours.print(Colours.RED, "skipping plot because of bad data: ", end="")
        print(', '.join([str(x.val) for x in result.invars]))
        return True
    if not len(result.outvars):
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

    # Get the raw data.
    inittimes, buildtimes, preptimes, swaptimes, skeltimes, conttimes = _gettimes(result.outvars)

    # Process data.
    data = [
        ("init", describe(inittimes)),
        ("build", describe(buildtimes)),
        ("prep", describe(preptimes)),
        ("upload", describe(conttimes["ul"])),
        ("run", describe(skeltimes)),
        ("swap", describe(swaptimes)),
        ("download", describe(conttimes["dl"])),
    ]

    # Create plottable data.
    Y, Yerr, Labels = zip(*[(x[1][0], x[1][1], x[0]) for x in data])
    X = np.arange(len(Y))

    # Time subtotals.
    t_ninit = sum(Y[1:])
    t_nbuild = t_ninit - Y[1]
    t_dev = sum([x for x,y in zip(Y,Labels) if search("(upload|run|download)", y)])

    title = ', '.join([str(x.val) for x in invars])
    caption = ("Times: no-init: \\textbf{{{ninit:.2f}}} ms, "
               "no-build: \\textbf{{{nbuild:.2f}}} ms. "
               "device: \\textbf{{{dev:.2f}}} ms. "
               "\n"
               "ID: \\texttt{{{id}}}. {n} samples."
               .format(ninit=t_ninit,
                       nbuild=t_nbuild,
                       dev=t_dev,
                       id=resultscache.id(invars),
                       n=len(inittimes)))

    # Plot the data.
    width = 1
    plt.bar(X, Y, width, yerr=Yerr, ecolor='k', color=[
        '#777777', 'yellow', 'red', 'yellow', 'green', 'red', 'yellow'
    ])

    # Use LaTeX text rendering.
    fontsize=16
    rc('text', usetex=True)
    rc('font', size=fontsize)

    ax = plt.axes()

    #  Vertical major gridlines.
    ax.yaxis.grid(b=True, which='major', color="#aaaaaa", linestyle='-')

    # Set the graph bounds.
    plt.gca().set_position((.12, # Left padding
                            .24, # Bottom padding
                            .85, # Width
                            .65)) # Height

    # Axis text and limits.
    plt.ylabel('Time (ms)')
    plt.ylim(ymin=0) # Time is always positive.
    plt.title('\n'.join(wrap(title, 60)), fontsize=fontsize, weight="bold")
    plt.xticks(X + width / 2., Labels, rotation=90)
    plt.figtext(.02, .02, caption)

    # Finish up.
    _finalize(result, name)
