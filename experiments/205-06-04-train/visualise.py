#!/usr/bin/env python2

from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

import labm8 as lab
from labm8 import io
from labm8 import fs
from labm8 import math as labmath

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db

import experiment

def create_oracle_wgsizes_heatmaps(db):
    fs.mkdir("img/oracle/devices")
    fs.mkdir("img/oracle/kernels")
    fs.mkdir("img/oracle/datasets")

    space = db.oracle_param_space()
    space.heatmap("img/oracle/heatmap.png",
                  title="All")

    for i,device in enumerate(db.devices):
        io.info("Device heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE device='{0}')"
                 .format(device))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/devices/{0}.png"
                      .format(i), title=device)

    for kernel,ids in db.lookup_named_kernels().iteritems():
        io.info("Kernel heatmap", kernel, "...")
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN (SELECT id FROM scenarios WHERE "
                 "kernel IN ({0}))".format(",".join(id_wrapped)))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/kernels/{0}.png"
                      .format(kernel), title=kernel)

    for i,dataset in enumerate(db.datasets):
        io.info("Dataset heatmap", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE dataset='{0}')"
                 .format(dataset))
        space = db.oracle_param_space(where=where)
        space.heatmap("img/oracle/datasets/{0}.png"
                      .format(i), title=dataset)


def create_max_wgsizes_heatmaps(db):
    space = db.max_wgsize_space()
    space.heatmap("img/max_wgsizes.png",
                  title="Distribution of maximum workgroup sizes")


def create_coverage_reports(db):
    fs.mkdir("img/coverage/devices")
    fs.mkdir("img/coverage/kernels")
    fs.mkdir("img/coverage/datasets")

    fs.mkdir("img/safety/devices")
    fs.mkdir("img/safety/kernels")
    fs.mkdir("img/safety/datasets")

    # Per-device
    for i,device in enumerate(db.devices):
        io.info("Device coverage", i, "...")
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/devices/{0}.png"
                      .format(i), title=device, vmin=0, vmax=1)
        io.info("Device safety", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/devices/{0}.png"
                      .format(i), title=device, vmin=0, vmax=1)

        # Real benchmarks
        io.info("Device coverage, real", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE device='{0}')"
                 "AND scenario IN "
                 "(SELECT id FROM scenarios WHERE kernel IN "
                 "(SELECT id FROM kernel_names WHERE synthetic=0))"
                 .format(device))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/devices/{0}_real.png"
                      .format(i), title=device + ", real",
                      vmin=0, vmax=1)
        io.info("Device safety, real", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/devices/{0}_real.png"
                      .format(i), title=device + ", real",
                      vmin=0, vmax=1)

        # Synthetic benchmarks
        io.info("Device coverage, synthetic", i, "...")
        where = ("scenario IN "
                 "(SELECT id FROM scenarios WHERE device='{0}')"
                 "AND scenario IN "
                 "(SELECT id FROM scenarios WHERE kernel IN "
                 "(SELECT id FROM kernel_names WHERE synthetic=1))"
                 .format(device))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/devices/{0}_synthetic.png"
                      .format(i), title=device + ", synthetic",
                      vmin=0, vmax=1)
        io.info("Device safety, synthetic", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/devices/{0}_synthetic.png"
                      .format(i), title=device + ", synthetic",
                      vmin=0, vmax=1)

    # Per-dataset
    for i,dataset in enumerate(db.datasets):
        io.info("Dataset coverage", i, "...")
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/datasets/{0}.png"
                      .format(i), title=dataset, vmin=0, vmax=1)
        io.info("Device safety", i, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/datasets/{0}.png"
                      .format(i), title=dataset, vmin=0, vmax=1)

    # Per-kerenl
    for kernel,ids in db.lookup_named_kernels().iteritems():
        io.info("Kernel coverage", kernel, "...")
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        space = db.param_coverage_space(where=where)
        space.heatmap("img/coverage/kernels/{0}.png"
                      .format(kernel), title=kernel, vmin=0, vmax=1)
        io.info("Kernel safety", kernel, "...")
        space = db.param_safe_space(where=where)
        space.heatmap("img/safety/kernels/{0}.png"
                      .format(kernel), title=kernel, vmin=0, vmax=1)


def create_params_plot(db):
    io.info("Plotting params performance ...")
    summary = db.params_summary()
    X = np.arange(len(summary))
    Labels = [t[0] for t in summary]
    Performance = [t[1] * 100 for t in summary]
    Coverage = [t[2] * 100 for t in summary]
    ax = plt.subplot(111)
    ax.plot(X, Performance, 'b', label="Performance")
    ax.plot(X, Coverage, 'g', label="Legality")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tight_layout()
    plt.legend()
    plt.savefig("img/params.png")
    plt.close()


def create_performance_plots(db):
    # Performance of all params across kernels.
    io.info("Plotting kernels performance ...")
    names = db.kernel_names
    Y = [db.performance_of_kernels_with_name(name) for name in names]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(names, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across kernels")
    plt.savefig("img/kernel_performance.png")
    plt.close()

    # Performance of all params across devices.
    io.info("Plotting devices performance ...")
    devices = db.cpus + db.gpus
    Y = [db.performance_of_device(device) for device in devices]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(devices, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across devices")
    plt.savefig("img/device_performance.png")
    plt.close()

    # Performance of all params across dataset.
    io.info("Plotting datasets performance ...")
    datasets = db.datasets
    Y = [db.performance_of_dataset(dataset) for dataset in datasets]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.boxplot(Y)
    ax.set_xticklabels(datasets, rotation=90)
    plt.ylim(ymin=0,ymax=1)
    plt.title("Workgroup size performance across datasets")
    plt.savefig("img/dataset_performance.png")
    plt.close()


def create_maxspeedups_plots(db):
    io.info("Plotting max speedups ...")
    Speedups = sorted(db.max_speedups().values())
    X = np.arange(len(Speedups))
    plt.plot(X, Speedups, 'b')
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off') # labels along the bottom edge are off
    plt.tight_layout()
    plt.savefig("img/max_speedups.png")
    plt.close()


def create_min_max_plot(db):
    io.info("Plotting min max runtimes ...")
    data = [t[2:] for t in db.min_max_runtimes()]
    min_t, max_t = zip(*data)

    iqr = (0.25, 0.75) # IQR to plot.
    nbins = 25 # Number of bins.

    lower = labmath.filter_iqr(min_t, *iqr)
    upper = labmath.filter_iqr(max_t, *iqr)

    min_data = np.r_[lower, upper].min()
    max_data = np.r_[lower, upper].max()
    bins = np.linspace(min_data, max_data, nbins)

    plt.hist(lower, bins, label="Min")
    plt.hist(upper, bins, label="Max");
    plt.title("Normalised distribution of min and max runtimes")
    plt.ylabel("Frequency")
    plt.xlabel("Runtime (normalised to mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("img/min_max_runtimes.png")
    plt.close()


def create_num_samples_plot(db):
    io.info("Plotting num samples ...")
    data = sorted([t[2] for t in db.num_samples()])

    nbins = 25 # Number of bins.

    bins = np.linspace(min(data), max(data), nbins)
    plt.hist(data, bins)
    plt.title("Sample counts for unique scenarios and params")
    plt.ylabel("Frequency")
    plt.xlabel("Number of samples")
    plt.tight_layout()
    plt.savefig("img/num_samples.png")
    plt.close()


def main():
    db = _db.Database(experiment.ORACLE_PATH)

    # Delete any old stuff.
    fs.rm("img")
    fs.mkdir("img")

    create_num_samples_plot(db)
    create_min_max_plot(db)
    create_maxspeedups_plots(db)
    create_performance_plots(db)
    create_params_plot(db)
    create_coverage_reports(db)
    create_oracle_wgsizes_heatmaps(db)
    create_max_wgsizes_heatmaps(db)


if __name__ == "__main__":
    main()
