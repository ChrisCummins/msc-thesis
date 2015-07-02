#!/usr/bin/env python2

from __future__ import division

import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

import labm8 as lab
from labm8 import io
from labm8 import fmt
from labm8 import fs
from labm8 import math as labmath
from labm8 import text

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db
from omnitune.skelcl import visualise
from omnitune.skelcl import space as _space

import experiment


def main():
    db = _db.Database(experiment.ORACLE_PATH)

    # Delete any old stuff.
    fs.rm("img")

    # Make directories
    fs.mkdir("img/scenarios/")

    fs.mkdir("img/eval/classifiers")
    fs.mkdir("img/eval/err_fn")

    fs.mkdir("img/coverage/devices")
    fs.mkdir("img/coverage/kernels")
    fs.mkdir("img/coverage/datasets")

    fs.mkdir("img/safety/devices")
    fs.mkdir("img/safety/kernels")
    fs.mkdir("img/safety/datasets")

    fs.mkdir("img/oracle/devices")
    fs.mkdir("img/oracle/kernels")
    fs.mkdir("img/oracle/datasets")

    visualise.xval_classification(db, "img/eval/xval_classification.png")

    # ML visualisations
    for i,classifier in enumerate(db.classifiers):
        visualise.xval_classifier_speedups(db, classifier,
                                           "img/eval/classifiers/{}.png"
                                           .format(i))

    visualise.xval_classifiers_accuracy(db, "img/eval/accuracy.png")
    visualise.xval_classifiers_invalid(db, "img/eval/invalid.png")

    # ML results table
    job = "xval_classifiers"
    query = db.execute(
        "SELECT classifier,err_fn,Count(*) AS count\n"
        "FROM classification_results\n"
        "WHERE job=? GROUP BY classifier,err_fn",
        (job,)
    )
    results = []
    for classifier,err_fn,count in query:
        correct, invalid, performance, speedup = zip(*[
            row for row in db.execute(
                "SELECT correct,invalid,performance,speedup\n"
                "FROM classification_results\n"
                "WHERE job=? AND classifier=? AND err_fn=?",
                (job, classifier, err_fn)
            )
        ])
        results.append([
            classifier,
            err_fn,
            (sum(correct) / count) * 100,
            (sum(invalid) / count) * 100,
            min(performance) * 100,
            labmath.geomean(performance) * 100,
            max(performance) * 100,
            min(speedup),
            labmath.geomean(speedup),
            max(speedup)
        ])

    str_args = {
        "float_format": lambda f: "{:.2f}".format(f)
    }

    print(fmt.table(results, str_args, columns=(
        "CLASSIFIER",
        "ERR_FN",
        "ACC %",
        "INV %",
        "Omin %",
        "Oavg %",
        "Omax %",
        "Smin",
        "Savg",
        "Smax",
    )))

    # Whole-dataset plots
    visualise.sample_counts(db, "img/sample_counts.png")
    visualise.runtimes_range(db, "img/runtimes_range.png")
    visualise.max_speedups(db, "img/max_speedups.png")
    visualise.kernel_performance(db, "img/kernel_performance.png")
    visualise.device_performance(db, "img/device_performance.png")
    visualise.dataset_performance(db, "img/dataset_performance.png")
    visualise.num_params_vs_accuracy(db, "img/num_params_vs_accuracy.png")
    visualise.performance_vs_coverage(db, "img/performance_vs_coverage.png")
    visualise.performance_vs_max_wgsize(db, "img/performance_vs_max_wgsize.png")
    visualise.max_wgsizes(db, "img/max_wgsizes.png")

    visualise.coverage(db, "img/coverage/coverage.png")
    visualise.safety(db, "img/safety/safety.png")
    visualise.oracle_wgsizes(db, "img/oracle/all.png")

    # Per-scenario plots
    for row in db.scenario_properties:
        scenario,device,kernel,north,south,east,west,width,height,tout = row
        output = "img/scenarios/{id}.png".format(id=scenario)
        title = ("{device}: {kernel}[{n},{s},{e},{w}]\n"
                 "{width} x {height} {type}s"
                 .format(device=text.truncate(device, 18), kernel=kernel,
                         n=north, s=south, e=east, w=west,
                         width=width, height=height, type=tout))

        visualise.scenario_performance(db, scenario, output, title=title)

    # Per-device plots
    for i,device in enumerate(db.devices):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        output = "img/coverage/devices/{0}.png".format(i)
        visualise.coverage(db, output=output, where=where, title=device)
        output = "img/safety/devices/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=device)
        output = "img/oracle/devices/{0}.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where, title=device)

        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=0\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_real.png".format(i)
        visualise.coverage(db, output=output, where=where,
                           title=device + ", real")
        output = "img/safety/devices/{0}_real.png".format(i)
        visualise.safety(db, output, where=where,
                         title=device + ", real")
        output = "img/oracle/devices/{0}_real.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where,
                                 title=device + ", real")


        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=1\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = "img/coverage/devices/{0}_synthetic.png".format(i)
        visualise.coverage(db, output=output, where=where,
                           title=device + ", synthetic")
        output = "img/safety/devices/{0}_synthetic.png".format(i)
        visualise.safety(db, output, where=where,
                         title=device + ", synthetic")
        output = "img/oracle/devices/{0}_synthetic.png".format(i)
        visualise.oracle_wgsizes(db, output, where=where,
                                 title=device + ", synthetic")

    # Per-kernel plots
    for kernel,ids in db.lookup_named_kernels().iteritems():
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        output = "img/coverage/kernels/{0}.png".format(kernel)
        visualise.coverage(db, output=output, where=where, title=kernel)
        output = "img/safety/kernels/{0}.png".format(kernel)
        visualise.safety(db, output=output, where=where, title=kernel)
        output = "img/oracle/kernels/{0}.png".format(kernel)
        visualise.safety(db, output=output, where=where, title=kernel)

    # Per-dataset plots
    for i,dataset in enumerate(db.datasets):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        output = "img/coverage/datasets/{0}.png".format(i)
        visualise.coverage(db, output, where=where, title=dataset)
        output = "img/safety/datasets/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=dataset)
        output = "img/oracle/datasets/{0}.png".format(i)
        visualise.safety(db, output, where=where, title=dataset)


if __name__ == "__main__":
    main()
