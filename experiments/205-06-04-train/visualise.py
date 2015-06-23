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
from omnitune.skelcl import visualise

import experiment


def create_oracle_wgsizes_heatmaps(db):
    fs.mkdir("img/oracle/devices")
    fs.mkdir("img/oracle/kernels")
    fs.mkdir("img/oracle/datasets")

    space = db.oracle_param_space()
    space.heatmap("img/oracle/heatmap.png",
                  title="All data")

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

    io.info("Coverage ...")
    space = db.param_coverage_space()
    space.heatmap("img/coverage/coverage.png",
                  title="All data", vmin=0, vmax=1)

    io.info("Safety ...")
    space = db.param_safe_space()
    space.heatmap("img/safety/safety.png",
                  title="All data", vmin=0, vmax=1)

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


def create_perf_coverage_plot(db):
    io.info("Plotting average params performance as function of legality ...")
    data = sorted([
        (
            db.perf_param_avg(param) * 100,
            db.perf_param_avg_legal(param) * 100,
            db.param_coverage(param) * 100
        )
        for param in db.params
    ], reverse=True, key=lambda x: (x[0], x[2], x[1]))
    X = np.arange(len(data))

    GeoPerformance, Performance, Coverage = zip(*data)

    ax = plt.subplot(111)
    ax.plot(X, Coverage, 'r', label="Legality")
    ax.plot(X, Performance, 'g', label="Performance (when legal)")
    ax.plot(X, GeoPerformance, 'b', label="Performance")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.title("Workgroup size performance vs. legality")
    plt.ylabel("Performance / Legality")
    plt.xlabel("Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    plt.savefig("img/perf_coverage.png")
    plt.close()


def create_perf_max_wgsize(db):
    io.info("Plotting params performance as function of max wgsize ...")
    data = sorted([
        (
            db.perf(scenario, param) * 100,
            db.ratio_max_wgsize(scenario, param) * 100
        )
        for scenario, param in db.scenario_params
    ], reverse=True)
    X = np.arange(len(data))

    Performance, Ratios = zip(*data)

    ax = plt.subplot(111)
    ax.plot(X, Ratios, 'g', label="Ratio max wgsize")
    ax.plot(X, Performance, 'b', label="Performance")
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d%%'))
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=100)
    plt.title("Workgroup size performance vs. maximum workgroup size")
    plt.ylabel("Performance / Ratio max wgsize")
    plt.xlabel("Scenarios, Parameters")
    plt.tight_layout()
    plt.legend(frameon=True)
    plt.savefig("img/perf_max_wgisze.png")
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
    Speedups = sorted(db.max_speedups().values(), reverse=True)
    X = np.arange(len(Speedups))
    plt.plot(X, Speedups, 'b')
    plt.xlim(xmin=0, xmax=len(X) - 1)
    plt.ylim(ymin=0, ymax=10)
    plt.axhline(y=1, color="k")
    plt.title("Max attainable speedups")
    plt.ylabel("Max speedup")
    plt.xlabel("Scenarios")
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
    plt.legend(frameon=True)
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

    visualise.num_samples(db, "img/num_samples.png")
    visualise.min_max_runtimes(db, "img/min_max_runtimes.png")
    visualise.max_speedups(db, "img/max_speedups.png")
    visualise.kernel_performance(db, "img/kernel_performance.png")
    visualise.device_performance(db, "img/device_performance.png")
    visualise.dataset_performance(db, "img/dataset_performance.png")

    visualise.performance_vs_coverage(db, "img/performance_vs_coverage.png")
    visualise.performance_vs_max_wgsize(db, "img/performance_vs_max_wgsize.png")
    visualise.max_wgsizes_heatmap(db, "img/max_wgsizes.png")
    # create_oracle_wgsizes_heatmaps(db)
    # create_max_wgsizes_heatmaps(db)

    fs.mkdir("img/coverage/devices")
    fs.mkdir("img/coverage/kernels")
    fs.mkdir("img/coverage/datasets")

    fs.mkdir("img/safety/devices")
    fs.mkdir("img/safety/kernels")
    fs.mkdir("img/safety/datasets")

    visualise.coverage(db, "img/coverage/coverage.png")
    visualise.safety(db, "img/safety/safety.png")

    # Per-device
    for i,device in enumerate(db.devices):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        output = "img/coverage/devices/{0}.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=device)

        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=0\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_real.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}_real.png".format(i)
        visualise.safety(db, output, where=where, title=device)

        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=1\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_synthetic.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}_synthetic.png".format(i)
        visualise.safety(db, output, where=where, title=device)

    # Per-kernel
    for kernel,ids in db.lookup_named_kernels().iteritems():
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        output = "img/coverage/kernels/{0}.png".format(kernel)
        visualise.coverage(db, output=output, where=where, title=kernel)
        output = "img/safety/kernels/{0}.png".format(kernel)
        visualise.safety(db, output=output, where=where, title=kernel)

    # Per-dataset
    for i,dataset in enumerate(db.datasets):
        io.info("Dataset coverage", i, "...")
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        output = "img/coverage/datasets/{0}.png".format(i)
        visualise.coverage(db, output, where=where, title=dataset)
        output = "img/safety/datasets/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=dataset)


if __name__ == "__main__":
    main()
