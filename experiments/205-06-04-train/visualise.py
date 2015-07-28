#!/usr/bin/env python2

from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter

import labm8 as lab
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import latex
from labm8 import math as labmath
from labm8 import ml
from labm8 import text
from labm8 import viz

from eval import Dataset

import omnitune
from omnitune import skelcl
from omnitune.skelcl import db as _db
from omnitune.skelcl import space as _space
from omnitune.skelcl import visualise
from omnitune.skelcl import unhash_params

import experiment


def features_tab(db, path):
    def _attribute_type(attribute):
        if attribute.type == 0:
            return "numeric"
        else:
            return "categorical: {}".format(attribute.num_values)

    def _format_name_col(name):
        return "\\texttt{{{}}}".format(latex.escape(name))

    def _table(rows, output):
        latex.table(rows, output=output, columns=("Name", "Type"),
                    escape=False, formatters=(_format_name_col, None))

    db.dump_csvs("/tmp/omnitune/visualise")
    dataset = Dataset.load("/tmp/omnitune/visualise/oracle_params.csv", db)

    attributes = [[attribute.name, _attribute_type(attribute)]
                  for attribute in dataset.instances.attributes()]
    features = attributes[1:-1]
    half = labmath.ceil(len(features) / 2)

    _table(features[:half], fs.path(path, "features.1.tex"))
    _table(features[half:], fs.path(path, "features.2.tex"))


def visualise_classification_job(db, job):
    basedir = "img/classification/{}/".format(job)

    fs.mkdir(basedir)
    fs.mkdir(basedir + "classifiers")
    fs.mkdir(basedir + "err_fns")

    visualise.err_fn_performance(db, basedir + "err_fns.png", job=job)

    # Bar plot of all results.
    visualise.classification(db, "img/classification/{}.png".format(job),
                             job=job)

    # Per-classifier plots.
    for i,classifier in enumerate(db.classification_classifiers):
        visualise.classifier_speedups(db, classifier,
                                      basedir + "classifiers/{}.png".format(i),
                                      job=job)
    # Per-err_fn plots.
    for err_fn in db.err_fns:
        visualise.err_fn_speedups(db, err_fn,
                                  basedir + "err_fns/{}.png".format(err_fn),
                                  job=job, sort=True)

    # Results table.
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

    for i in range(len(results)):
        results[i][0] = ml.classifier_basename(results[i][0])

    columns=(
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
    )

    latex.table(results, output=fs.path(experiment.TAB_ROOT, job + ".tex"),
                columns=columns, **str_args)


def visualise_regression_job(db, job):
    runtimedir = "img/runtime_regression/{}/".format(job)
    runtimeclassificationdir = "img/runtime_classification/{}/".format(job)
    speedupdir = "img/speedup_regression/{}/".format(job)
    speedupclassificationdir = "img/speedup_classification/{}/".format(job)

    fs.mkdir(runtimedir)
    fs.mkdir(runtimeclassificationdir)
    fs.mkdir(speedupdir)
    fs.mkdir(speedupclassificationdir)

    # Line plot of all classifiers.
    visualise.runtime_regression(db,
                                 "img/runtime_regression/{}.png".format(job),
                                 job=job)
    visualise.runtime_classification(db,
                                     "img/runtime_classification/{}.png"
                                     .format(job), job=job)
    visualise.speedup_regression(db,
                                 "img/speedup_regression/{}.png".format(job),
                                 job=job)
    visualise.speedup_classification(db,
                                     "img/speedup_classification/{}.png"
                                     .format(job), job=job)


def main():
    db = _db.Database(experiment.ORACLE_PATH)
    ml.start()

    # Delete any old stuff.
    fs.rm(experiment.IMG_ROOT + "/*")
    fs.rm(experiment.TAB_ROOT + "/*")

    # Make directories
    fs.mkdir(experiment.TAB_ROOT)
    fs.mkdir(fs.path(experiment.IMG_ROOT, "scenarios/bars"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "scenarios/heatmap"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "scenarios/trisurf"))

    fs.mkdir(fs.path(experiment.IMG_ROOT, "coverage/devices"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "coverage/kernels"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "coverage/datasets"))

    fs.mkdir(fs.path(experiment.IMG_ROOT, "safety/devices"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "safety/kernels"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "safety/datasets"))

    fs.mkdir(fs.path(experiment.IMG_ROOT, "oracle/devices"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "oracle/kernels"))
    fs.mkdir(fs.path(experiment.IMG_ROOT, "oracle/datasets"))

    visualise.pie(db.num_scenarios_by_device,
                  fs.path(experiment.IMG_ROOT, "num_sceanrios_by_device"))
    visualise.pie(db.num_runtime_stats_by_device,
                  fs.path(experiment.IMG_ROOT, "num_runtime_stats_by_device"))
    visualise.pie(db.num_scenarios_by_dataset,
                  fs.path(experiment.IMG_ROOT, "num_sceanrios_by_dataset"))
    visualise.pie(db.num_runtime_stats_by_dataset,
                  fs.path(experiment.IMG_ROOT, "num_runtime_stats_by_dataset"))
    visualise.pie(db.num_runtime_stats_by_kernel,
                  fs.path(experiment.IMG_ROOT, "num_runtime_stats_by_kernel"))
    visualise.pie(db.num_runtime_stats_by_kernel,
                  fs.path(experiment.IMG_ROOT, "num_runtime_stats_by_kernel"))

    # Per-scenario plots
    for row in db.scenario_properties:
        scenario,device,kernel,north,south,east,west,max_wgsize,width,height,tout = row
        title = ("{device}: {kernel}[{n},{s},{e},{w}]\n"
                 "{width} x {height} {type}s"
                 .format(device=text.truncate(device, 18), kernel=kernel,
                         n=north, s=south, e=east, w=west,
                         width=width, height=height, type=tout))
        output = fs.path(experiment.IMG_ROOT,
                         "scenarios/heatmap/{id}.png".format(id=scenario))
        space = _space.ParamSpace.from_dict(db.perf_scenario(scenario))
        max_c = min(25, len(space.c))
        max_r = min(25, len(space.r))
        space.reshape(max_c=max_c, max_r=max_r)

        # Heatmaps.
        mask = _space.ParamSpace(space.c, space.r)
        for j in range(len(mask.r)):
            for i in range(len(mask.c)):
                if space.matrix[j][i] == 0:
                    r, c = space.r[j], space.c[i]
                    # TODO: Get values from refused_params table.
                    if r * c >= max_wgsize:
                        mask.matrix[j][i] = -1
                    else:
                        mask.matrix[j][i] = 1

        new_order = list(reversed(range(space.matrix.shape[0])))
        data = space.matrix[:][new_order]

        figsize=(12,6)

        _, ax = plt.subplots(1, 2, figsize=figsize, sharey=True)
        sns.heatmap(data, ax=ax[0],
                    xticklabels=space.c,
                    yticklabels=list(reversed(space.r)), square=True)

        ax[0].set_title(title)

        new_order = list(reversed(range(mask.matrix.shape[0])))
        data = mask.matrix[:][new_order]

        sns.heatmap(data, ax=ax[1], vmin=-1, vmax=1,
                    xticklabels=space.c,
                    yticklabels=list(reversed(space.r)), square=True)

        # Set labels.
        ax[0].set_ylabel("Rows")
        ax[0].set_xlabel("Columns")
        ax[1].set_ylabel("Rows")
        ax[1].set_xlabel("Columns")

        # plt.tight_layout()
        # plt.gcf().set_size_inches(*figsize, dpi=300)

        viz.finalise(output)

        # 3D bars.
        output = fs.path(experiment.IMG_ROOT,
                         "scenarios/bars/{id}.png".format(id=scenario))
        space.bar3d(output=output, title=title, zlabel="Performance",
                    rotation=45)

        # Trisurfs.
        output = fs.path(experiment.IMG_ROOT,
                         "scenarios/trisurf/{id}.png".format(id=scenario))
        space.trisurf(output=output, title=title, zlabel="Performance",
                      rotation=45)

    # Per-device plots
    for i,device in enumerate(db.devices):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE device='{0}')"
                 .format(device))
        output = fs.path(experiment.IMG_ROOT,
                         "coverage/devices/{0}.png".format(i))
        visualise.coverage(db, output=output, where=where, title=device)
        output = fs.path(experiment.IMG_ROOT,
                         "safety/devices/{0}.png".format(i))
        visualise.safety(db, output, where=where, title=device)
        output = fs.path(experiment.IMG_ROOT,
                         "oracle/devices/{0}.png".format(i))
        visualise.oracle_wgsizes(db, output, where=where, title=device)

        where = ("scenario IN (\n"
                 "    SELECT id from scenarios WHERE device='{0}'\n"
                 ") AND scenario IN (\n"
                 "    SELECT id FROM scenarios WHERE kernel IN (\n"
                 "        SELECT id FROM kernel_names WHERE synthetic=0\n"
                 "    )\n"
                 ")"
                 .format(device))
        output = fs.path(experiment.IMG_ROOT,
                         "coverage/devices/{0}_real.png".format(i))
        visualise.coverage(db, output=output, where=where,
                           title=device + ", real")
        output = fs.path(experiment.IMG_ROOT,
                         "safety/devices/{0}_real.png".format(i))
        visualise.safety(db, output, where=where,
                         title=device + ", real")
        output = fs.path(experiment.IMG_ROOT,
                         "oracle/devices/{0}_real.png".format(i))
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
        output = fs.path(experiment.IMG_ROOT,
                         "coverage/devices/{0}_synthetic.png".format(i))
        visualise.coverage(db, output=output, where=where,
                           title=device + ", synthetic")
        output = fs.path(experiment.IMG_ROOT,
                         "safety/devices/{0}_synthetic.png".format(i))
        visualise.safety(db, output, where=where,
                         title=device + ", synthetic")
        output = fs.path(experiment.IMG_ROOT,
                         "oracle/devices/{0}_synthetic.png".format(i))
        visualise.oracle_wgsizes(db, output, where=where,
                                 title=device + ", synthetic")

    # Per-kernel plots
    for kernel,ids in db.lookup_named_kernels().iteritems():
        id_wrapped = ['"' + id + '"' for id in ids]
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE kernel IN ({0}))"
                 .format(",".join(id_wrapped)))
        output = fs.path(experiment.IMG_ROOT,
                         "coverage/kernels/{0}.png".format(kernel))
        visualise.coverage(db, output=output, where=where, title=kernel)
        output = fs.path(experiment.IMG_ROOT,
                         "safety/kernels/{0}.png".format(kernel))
        visualise.safety(db, output=output, where=where, title=kernel)
        output = fs.path(experiment.IMG_ROOT,
                         "oracle/kernels/{0}.png".format(kernel))
        visualise.safety(db, output=output, where=where, title=kernel)

    # Per-dataset plots
    for i,dataset in enumerate(db.datasets):
        where = ("scenario IN "
                 "(SELECT id from scenarios WHERE dataset='{0}')"
                 .format(dataset))
        output = fs.path(experiment.IMG_ROOT,
                         "coverage/datasets/{0}.png".format(i))
        visualise.coverage(db, output, where=where, title=dataset)
        output = fs.path(experiment.IMG_ROOT,
                         "safety/datasets/{0}.png".format(i))
        visualise.safety(db, output, where=where, title=dataset)
        output = fs.path(experiment.IMG_ROOT,
                         "oracle/datasets/{0}.png".format(i))
        visualise.safety(db, output, where=where, title=dataset)

    #####################
    # ML Visualisations #
    #####################
    features_tab(db, experiment.TAB_ROOT)

    visualise_classification_job(db, "xval")
    visualise_classification_job(db, "arch")
    visualise_classification_job(db, "xval_real")
    visualise_classification_job(db, "synthetic_real")

    # Runtime regression accuracy.
    visualise_regression_job(db, "xval")
    visualise_regression_job(db, "arch")
    visualise_regression_job(db, "xval_real")
    visualise_regression_job(db, "synthetic_real")

    # Whole-dataset plots
    visualise.runtimes_variance(db, fs.path(experiment.IMG_ROOT,
                                            "runtime_variance.png"),
                                min_samples=30)
    visualise.num_samples(db, fs.path(experiment.IMG_ROOT,
                                      "num_samples.png"))
    visualise.runtimes_range(db, fs.path(experiment.IMG_ROOT,
                                         "runtimes_range.png"))
    visualise.max_speedups(db, fs.path(experiment.IMG_ROOT,
                                       "max_speedups.png"))
    visualise.kernel_performance(db, fs.path(experiment.IMG_ROOT,
                                             "kernel_performance.png"))
    visualise.device_performance(db, fs.path(experiment.IMG_ROOT,
                                             "device_performance.png"))
    visualise.dataset_performance(db, fs.path(experiment.IMG_ROOT,
                                              "dataset_performance.png"))
    visualise.num_params_vs_accuracy(db, fs.path(experiment.IMG_ROOT,
                                                 "num_params_vs_accuracy.png"))
    visualise.performance_vs_coverage(db,
                                      fs.path(experiment.IMG_ROOT,
                                              "performance_vs_coverage.png"))
    visualise.performance_vs_max_wgsize(
        db, fs.path(experiment.IMG_ROOT, "performance_vs_max_wgsize.png")
    )
    visualise.performance_vs_wgsize(db, fs.path(experiment.IMG_ROOT,
                                                "performance_vs_wgsize.png"))
    visualise.performance_vs_wg_c(db, fs.path(experiment.IMG_ROOT,
                                              "performance_vs_wg_c.png"))
    visualise.performance_vs_wg_r(db, fs.path(experiment.IMG_ROOT,
                                              "performance_vs_wg_r.png"))
    visualise.max_wgsizes(db, fs.path(experiment.IMG_ROOT, "max_wgsizes.png"))
    visualise.oracle_speedups(db, fs.path(experiment.IMG_ROOT,
                                          "oracle_speedups.png"))

    visualise.coverage(db,
                       fs.path(experiment.IMG_ROOT, "coverage/coverage.png"))
    visualise.safety(db, fs.path(experiment.IMG_ROOT, "safety/safety.png"))
    visualise.oracle_wgsizes(db, fs.path(experiment.IMG_ROOT, "oracle/all.png"))

    ml.stop()

if __name__ == "__main__":
    main()
