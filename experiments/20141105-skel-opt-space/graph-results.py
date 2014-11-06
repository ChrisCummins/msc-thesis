#!/usr/bin/env python

from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.mlab import griddata
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import sys

if len(sys.argv) != 3:
    print("Usage: graph-results.py <dataset> <n>")
    sys.exit(1)

r = mlab.csv2rec(sys.argv[1], delimiter=' ')
n = sys.argv[2]

x = []
for i in r:
    x.append(i[1])
X = np.array(x)
y = []
for i in r:
    y.append(i[0])
Y = np.array(y)
z = []
for i in r:
    z.append(i[3])

# Convert all times into speedups relative to the first result.
scale = z[0]
for i in range(len(z)):
    z[i] = scale / z[i]

Z = np.array(z)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Speedup is between 1 and 4. Note this causes slowdowns to "fall
# through the floor".
ax.set_zlim(1, 8)

# vmin and vmax are the limits of the colour maps.
surf = ax.plot_trisurf(X, Y, Z, cmap=cm.jet, vmin=1.0, vmax=3.5, linewidth=.2)
plt.gca().invert_xaxis()

plt.suptitle('Optimisation space for merge sort, n=1e{0}'.format(n), fontsize=16)
plt.xlabel('Parallelisation depth')
plt.ylabel('Split threshold')
ax.set_zlabel('Speedup')
plt.savefig('plot.png')
plt.show()
