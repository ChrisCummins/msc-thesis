#!/usr/bin/env python

# Principle component analysis in (almost) pure Python.

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

# Return the dot product of two variables.
def dp(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def plot_3d(data):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10

    A = list(filter(lambda x: x[3] == "A", data))
    B = list(filter(lambda x: x[3] == "B", data))

    ax.plot([a[0] for a in A],
            [a[1] for a in A],
            [a[2] for a in A],
            'o', markersize=8, color='blue', alpha=0.5, label='A')
    ax.plot([b[0] for b in B],
            [b[1] for b in B],
            [b[2] for b in B],
            'o', markersize=8, color='red', alpha=0.5, label='B')

    plt.title('Samples for class 1 and class 2')
    ax.legend(loc='upper right')
    plt.show()
    plt.close()

def plot_2d(data):
    A = list(filter(lambda x: x[2] == "A", data))
    B = list(filter(lambda x: x[2] == "B", data))

    plt.plot([a[0] for a in A],
             [a[1] for a in A],
             'o', markersize=8, color='blue', alpha=0.5, label='A')
    plt.plot([b[0] for b in B],
             [b[1] for b in B],
             'o', markersize=8, color='red', alpha=0.5, label='B')
    plt.xlabel('x_values')
    plt.ylabel('y_values')
    plt.legend()
    plt.title('Transformed samples with class labels')

    plt.show()

# PCA to reduce a d-dimensional data set to a k-dimensional data set:
#####################################################################

# PCA reduces a d-dimensional dataset to a k-dimensional dataset. In
# this case, we're going to reduce a 3 dimensional set to 2
# dimensions. Our dataset consists of 5 labelled data points,
# belonging to class A or B.
X = [[4, 2, .6, "A"],
     [4.2, 2.1, .59, "A"],
     [3.9, 2, .58, "A"],
     [4.3, 2.1, .62, "B"],
     [4.1, 2.2, .63, "B"]]
# The length of our dataset is n.
n = len(X)
# The dimensionality of our dataset.
d = len(X[0]) - 1
# The reduced dimensionality size.
k = 2

plot_3d(X)

###
### Perform PCA:
###

# 1. Compute the mean vector. The mean vector has length d.
mean = [
    sum([x[0] for x in X]) / len(X),
    sum([x[1] for x in X]) / len(X),
    sum([x[2] for x in X]) / len(X)
]

# 2. "Centre" the dataset about the mean, by subtracting the mean
# vector from each data point.
for x in X:
    x[0] -= mean[0]
    x[1] -= mean[2]
    x[2] -= mean[2]

# 3. Compute the covariance matrix. The covariance matrix has size
# d x d. The main diagonal the covariance matrix shows is the variances
# of each attribute.
#
#    cov(a,b) = (1 / (n-1)) * sum(X[i][a] * X[i][b])
covariance = [[0,0,0],[0,0,0],[0,0,0]]
for i in range(d):
    for j in range(d):
        c = 0
        for l in range(n):
            c += X[l][i] * X[l][j]
        covariance[i][j] = c / (n - 1)

# 4. Compute the eigenvectors and eigenvalues.
eigen_values, eigen_vectors = np.linalg.eig(covariance)

# 5. Sort the eigenvectors by decreasing eigenvalues, and select the k
# eigenvectors with the highest eigenvalues.
e = list(reversed(sorted(zip(eigen_values, eigen_vectors),
                         key=lambda x: x[0])))
e = e[0:k]

# 6. Create k x d matrix of eigenvectors.
W = [x[1] for x in e]

# 7. Project the samples onto the k dimensional subspace:
#    x' = W^T * x
for i in range(n):
    X[i] = [
        dp([W[0][0], W[0][1], W[0][2]], X[i]),
        dp([W[1][0], W[1][1], W[1][2]], X[i]),
        X[i][3]
    ]

plot_2d(X)

print("Mean", mean)
print("Covariance", covariance)
print("W", W)
print("Reduced", X)
