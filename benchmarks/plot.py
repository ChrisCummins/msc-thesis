from __future__ import print_function
from hashlib import sha1
from textwrap import wrap
from matplotlib import rc

import matplotlib
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

def _writechecksum(path, checksum):
    file = open(path, 'a')
    file.write("<!-- __checksum__ {checksum} -->"
               .format(checksum=checksum))
    file.close()

_checksumre = compile("<!-- __checksum__ ([0-9a-f]+) -->")

def _readchecksum(path):
    # The checksum is embedded in the last line of image data, so we
    # employ a speedy hack to read the last line using seek(). If
    # anything fucks up, return None.
    #
    # See: http://stackoverflow.com/a/3346492/1318051
    with open(path, 'rb') as file:
        try:
            first = next(file).decode()

            file.seek(-1024, 2)
            last = file.readlines()[-1].decode()
            match = _checksumre.match(last)
            return match.group(1)
        except:
            return None

#
def _finalize(path=None):
    if path == None:
        plt.show()
    else:
        mkdir(dirname(path))
        Colours.print(Colours.BLUE, "Wrote {path} ...".format(path=path))
        plt.savefig(path)

    plt.close()


#
def _skippable(result, name):
    if result.bad:
        # Bad data is worth warning about.
        Colours.print(Colours.RED, "skipping plot because of bad data: ", end="")
        print(', '.join([str(x.val) for x in result.invars]))
        print('   ', resultscache.resultspath(result.invars))
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
    inittimes, buildtimes, preptimes, ultimes, skeltimes, swaptims, dltimes = gettimes(result.outvars)
    data = summarise(inittimes, buildtimes, preptimes, ultimes, skeltimes, swaptims, dltimes)

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
    path = resultscache.plotpath(result.invars, suffix="-{name}".format(name=name))
    _finalize(path)
    # Embed checksum of graphed data in the plot file.
    _writechecksum(path, _HashableResult(result).key())

def speedups(speedups, err=[], labels=[], xlabel="", ylabel="Speedup",
             title="", caption="", baseline=-1, ymajorlines=False, path=None):
    X = np.arange(len(speedups))

    # Plot the data.
    width = 1

    # Arguments to plt.bar()
    kwargs = {
        'width': width,
        'color': 'r'
    }

    # Add error bars, if provided.
    if err:
        kwargs['yerr'] = err
        kwargs['ecolor'] = 'k'

    plt.bar(X, speedups, **kwargs)

    ax = plt.axes()

    greenindexes = [x for x,v in enumerate(speedups) if v > 1]

    # Filter out the bars from a complex plot.
    bars = filter(lambda x: isinstance(x, matplotlib.patches.Rectangle), ax.get_children())

    # Colour the default value blue, and the speedups > 1 green.
    if baseline >= 0:
        bars[baseline].set_facecolor('b')

    for i in greenindexes:
        bars[i].set_facecolor('g')

    # Use LaTeX text rendering.
    fontsize=16
    rc('text', usetex=True)
    rc('font', size=fontsize)

    #  Vertical major gridlines.
    if ymajorlines:
        ax.yaxis.grid(b=True, which='major', color="#aaaaaa", linestyle='-')

    # Axis text and limits.
    plt.ylabel(ylabel)
    if xlabel:
        plt.xlabel(xlabel)

    plt.ylim(ymin=0) # Values are always positive.
    plt.xlim(xmin=0, xmax=X[-1] + 1) # Plot as much data as needed.
    plt.title('\n'.join(wrap(title, 60)), fontsize=fontsize, weight="bold")

    plt.axhline(y=1, color='k')

    if labels:
        plt.xticks(X + width / 2., labels, rotation=90, fontsize=10)

    position = [.12, # Left padding
                .14, # Bottom padding
                .85, # Width
                .75] # Height

    if caption:
        plt.figtext(.02, .02, caption)
        position[1] = .24 # Add extra padding to bottom.
        position[3] = .65

    # Set the graph bounds.
    plt.gca().set_position(position)

    # Finish up.
    _finalize(path)
