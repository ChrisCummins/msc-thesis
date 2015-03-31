from math import sqrt
from scipy import stats

import scipy

# Return the mean value of a list of divisible numbers.
def mean(num):
    if len(num):
        return sum([float(x) for x in num]) / float(len(num))
    else:
        return 0

# Return the variance of a list of divisible numbers.
def variance(num):
    if len(num) > 1:
        m = mean(num)
        return sum([(x - m) ** 2 for x in num]) / (len(num) - 1)
    else:
        return 0

# Return the standard deviation of a list of divisible numbers.
def stdev(num):
    return sqrt(variance(num))

# Return the confidence interval of a list for a given confidence
def confinterval(l, c=0.95, n=30):
    # Only calculate confidence if we have:
    #
    #   * More than one datapoint.
    #   * If at least one of the datapoints is not zero.
    if len(l) > 1 and not all(x == 0 for x in l):
        scale = stdev(l) / sqrt(len(l))

        if len(l) < n:
            # For small values of n, use a t-distribution:
            c1, c2 = scipy.stats.t.interval(c, len(l) - 1, loc=mean(l), scale=scale)
        else:
            # For large values of n, use a normal (Gaussian) distribution:
            c1, c2 = scipy.stats.norm.interval(c, loc=mean(l), scale=scale)

        return c1, c2
    else:
        return 0, 0

# Return a tuple of the mean and yerr.
def describe(num, **kwargs):
    num = [float(x) for x in num] # Cast all to floating points.
    c = confinterval(num, **kwargs)
    return mean(num), c[1] - mean(num)
