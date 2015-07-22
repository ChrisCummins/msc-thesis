#!/usr/bin/env python2

import random
import sys
import os

import labm8
from labm8 import io
from labm8 import fs
from labm8 import make
from labm8 import system

import experiment


def run_example_prog(prog, args):
    """
    Run a SkelCL example program.

    Arguments:

        prog (str): The name of the program to run
        args (list of str): Any arguments
    """
    fs.cd(fs.path(experiment.EXAMPLES_BUILD, prog))
    cmd = ["./" + prog] + args
    cmd_str = " ".join(cmd)
    io.info("COMMAND:", io.colourise(io.Colours.RED, cmd_str))
    ret, _, _ = system.run(cmd, stdout=system.STDOUT, stderr=system.STDERR)
    if ret:
        system.echo(cmd_str, "/tmp/naughty.txt", append=True)

    return ret


def sample_gaussian_blur(devargs, iterations):
    run_example_prog("GaussianBlur", devargs + [
        "--iterations", str(iterations)
    ])


def sample_canny(devargs, iterations):
    run_example_prog("CannyEdgeDetection", devargs + [
        "--iterations", str(iterations)
    ])


def sample_fdtd(devargs, iterations):
    run_example_prog("FDTD", devargs + [
        "--resolution", str(iterations / 10)
    ])


def sample_gol(devargs, iterations):
    run_example_prog("GameOfLife", devargs + [
        "--iterations", str(iterations)
    ])


def sample_heat_equation(devargs, iterations):
    run_example_prog("HeatEquation", devargs + [
        "--iterations", str(iterations)
    ])


def sample_simplebig(args, iterations=250):
    """
    Run the SimpleBig program.
    """
    run_example_prog("SimpleBig", args + [
        "--iterations", str(iterations)
    ])


def run_real_benchmarks(iterations=250):
    """
    Sample the space of real benchmarks.
    """
    for devargs in experiment.DEVARGS:
        sample_gaussian_blur(devargs, iterations)
        sample_canny(devargs, iterations)
        #sample_fdtd(devargs, iterations)
        sample_gol(devargs, iterations)
        sample_heat_equation(devargs, iterations)


def run_synthetic_benchmarks(iterations=250):
    """
    Sample the space of synthetic benchmarks.
    """
    allargs = list(experiment.SIMPLEBIG_ARGS)

    random.shuffle(allargs)

    for devargs in experiment.DEVARGS:
        for simplebigargs in allargs:
            args = labm8.flatten(simplebigargs + (devargs,))
            io.debug(" ".join(args))
            cmd_str = " ".join(args)

            sample_simplebig(args, iterations=iterations)


def sample_space(iterations=250):
    """
    Sample the space of all benchmarks.
    """
    run_synthetic_benchmarks(iterations=iterations)
    run_real_benchmarks(iterations=iterations)


def main():
    os.environ["OMNITUNE_OFFLINE_TRAINING"] = "1"
    fs.cd(experiment.EXAMPLES_BUILD)

    # Build sources.
    ret, _, _ = make.make()
    if ret:
        labm8.exit(ret)

    while True:
        sample_space()


if __name__ == "__main__":
    main()
