import srtime
from srtime.stats import Stats
from srtime.parser import ArgumentParser
from srtime.stats import Stats
import os
import sys

class Variable:
    def __iter__(self):
        return self

    def __next__(self):
        if self.has_next():
            return self.get_next()
        else:
            raise StopIteration

    def has_next(self):
        pass

    def get_next(self):
        return self.stats()

    def get_stats(self):
        pass

    def stats(self):
        pass

class RangeVariable(Variable):
    def __init__(self, start, end, step):
        self.start = start
        self.current = start
        self.end = end
        self.step = step
        # The "iterated" variable tracks whether
        # for i in RangeVariable(1, 3, 1)
        self.iterated = False

    def has_next(self):
        return self.current + self.step <= self.end

    def get_next(self):
        if self.iterated:
            self.current += self.step
        self.iterated = True
        return super().get_next()


def inplace_sed(pat, sub, file):
    os.system("sed -ri 's/{pat}/{sub}/' {file}"
              .format(pat=pat, sub=sub, file=file))


class InputSizeVariable(RangeVariable):

    def get_next(self):
        s = super().get_next()
        inplace_sed("(\\#define\\s+INPUT_SIZE\\s+).+",
                    "\\1{size}".format(size=self.current),
                    "stable_sort.cc")
        return s

    def stats(self):
        return [("size", self.current)]


class ThresholdVariable(RangeVariable):

    def get_next(self):
        s = super().get_next()
        inplace_sed("(\\#define\\s+THRESHOLD\\s+).+",
                    "\\1{size}".format(size=self.current),
                    "stable_sort.cc")
        return s

    def stats(self):
        return [("threshold", self.current)]


class ParDepthVariable(RangeVariable):

    def get_next(self):
        s = super().get_next()
        inplace_sed("(\\#define\\s+PAR_DEPTH\\s+).+",
                    "\\1{size}".format(size=self.current),
                    "stable_sort.cc")
        return s

    def stats(self):
        return [("par_depth", self.current)]


variables = [InputSizeVariable(int(1e5), int(1e6), int(2e4)),
             ThresholdVariable(2, 100, 10),
             ParDepthVariable(0, 2, 1)]

def flatten(lists):
    return [l for sublist in lists for l in sublist]

headers = False

def prepare_run():
    os.system("make stable_sort >/dev/null")

def do_run(args):
        args = ArgumentParser().parse_args(args)
        independent_variables = flatten([var.stats() for var in variables])
        dependent_variables = Stats(srtime.run(args))._attrs
        results = independent_variables + dependent_variables

        global headers
        if not headers:
            print(", ".join([str(i[0]) for i in results]), file=sys.stderr)
            headers = True
        print(", ".join([str(i[1]) for i in results]), file=sys.stderr)

for var in variables:
    for val in var:
        prepare_run()
        do_run(["./stable_sort", "-i", "-t", "2"])
