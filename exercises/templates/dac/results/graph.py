#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.mlab as mlab
import matplotlib.cm as cm
import sys
import statistics

r = mlab.csv2rec(sys.argv[1], delimiter=' ')
mean = []
n = r.n

fig = plt.figure()
ax1 = fig.add_subplot(111)

line_width = 1.5

# Set X axis range

colors = cm.rainbow(np.linspace(0, 1, len(r.dtype.names[1:])))

for series, color in zip(r.dtype.names[1:7], colors):

    # Create normalised dictionary of values:
    normalised = {}
    for x,y in zip(n, r[series]):
        if x in normalised:
            normalised[x].append(y)
        else:
            normalised[x] = [y]

    # Create array of mean value tuples:
    means = {}
    for x in normalised:
        means[x] = sum(normalised[x]) / float(len(normalised[x]))
    means = [(k, means[k]) for k in sorted(means)]

    # Create array of stdev tuples:
    stdevs = {}
    for x in normalised:
        stdevs[x] = statistics.stdev(normalised[x])
    stdevs = [(k, stdevs[k]) for k in sorted(stdevs)]
    
    x,y = zip(*means)
    x,err = zip(*stdevs)

    # Plot series and set line and error bar cap widths:
    (_, caps, _) = ax1.errorbar(x, y, yerr=err, label=series, color=color)
    ax1.lines[-1].set_linewidth(line_width)
    for cap in caps:
        cap.set_markeredgewidth(line_width)

plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.legend(loc='upper left');

xlim=(min(n), max(n))
ylim=(0, 200)

ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
plt.yticks(range(0, ylim[1], 50))
plt.xticks(range(max(xlim[0], 100000), xlim[1], 100000))

plt.tight_layout(pad=2.5)

plt.suptitle('Merge Sort with Divide and Conquer Skeleton', fontsize=16)
plt.ylabel('Execution time (ms)')
plt.xlabel('No of integers sorted')
plt.savefig('plot.png')
plt.show()

