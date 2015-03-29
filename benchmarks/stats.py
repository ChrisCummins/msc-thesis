from math import sqrt

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
    if len(l) > 1:
        scale = stdev(l) / sqrt(len(l))

        #if len(l) >= n:
            # For large values of n, use a normal (Gaussian) distribution:
            #c1, c2 = scipy.stats.norm.interval(c, loc=mean(l), scale=scale)
        #else:
            # For small values of n, use a t-distribution:
            #c1, c2 = scipy.stats.t.interval(c, len(l) - 1, loc=mean(l), scale=scale)
        return 0, 0
    else:
        return 0, 0