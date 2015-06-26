#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import re

from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from scipy import stats

import weka
from weka.classifiers import Classifier as WekaClassifier
from weka.classifiers import FilteredClassifier as WekaFilteredClassifier
from weka.filters import Filter as WekaFilter
from weka.core.classes import Random as WekaRandom
from weka.core.dataset import Instances

import labm8 as lab
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8.db import where

import omnitune
from omnitune.skelcl import db as _db
from omnitune.skelcl import hash_params
from omnitune.skelcl import unhash_params

import experiment

# Random number generator seed.
SEED = 0xcec


class Error(Exception):
    """
    Module-level error.
    """
    pass


class ErrFnError(Error):
    """
    Raised if an err_fn fails.
    """
    pass


class ReshapeError(ErrFnError):
    """
    Raised if the reshape_fn callback fails.
    """
    pass


def classifier2str(classifier):
    return " ".join([classifier.classname] + classifier.options)


def sanitise_classifier_str(classifier):
    return re.sub(r"[ -\.]+", "-", classifier)


def summarise_perfs(perfs):
    def _fmt(n):
        return "{:.3f}".format(n)

    print("        n: ", len(perfs))
    print("     mean: ", _fmt(labmath.mean(perfs)))
    print("  geomean: ", _fmt(labmath.geomean(perfs)))
    print("      min: ", _fmt(min(perfs)))
    print("      max: ", _fmt(max(perfs)))
    print()


class Dataset(ml.Dataset):

    def __init__(self, db, *args, **kwargs):
        super(Dataset, self).__init__(*args, **kwargs)
        self.db = db

    def one_r(self):
        pass

    @staticmethod
    def load(path, db):
        nominals = [
            49,  # dev_double_fp_config
            50,  # dev_endian_little
            51,  # dev_execution_capabilities
            52,  # dev_extensions
            54,  # dev_global_mem_cache_type
            57,  # dev_host_unified_memory
            63,  # dev_image_support
            65,  # dev_local_mem_type
            96,  # dev_queue_properties
            97,  # dev_single_fp_config
            98,  # dev_type
            100, # dev_vendor_id
        ]
        nominal_indices = ",".join([str(index) for index in nominals])
        force_nominal = ["-N", nominal_indices]

        dataset = ml.Dataset.load_csv(path, options=force_nominal)
        dataset.class_index = -1
        dataset.db = db

        return dataset


def perf_fn(db, baseline, scenario, predicted, oracle):
    """
    Performance callback.

    Arguments:

        db (skelcl.Database): Database object.
        baseline (str): Baseline params string to compare performance
          against.
        scenario (str): Scenario ID.
        predicted (str): Predicted parameters ID.
        oracle (str): Oracle parameters ID.

    Returns:

        (float, float): Ratio of performance to the oracle, and
          speedup over baseline params.
    """
    speedup = db.speedup(scenario, baseline, predicted)

    if predicted == oracle:
        return 1, speedup
    else:
        return db.perf(scenario, predicted), speedup


####################
# err_fn callbacks #
####################

def default_fn(default, *args, **kwargs):
    """
    Default value parameter callback.

    Simple returns the supplied default argument.
    """
    return default


def random_fn(db, instance, max_wgsize, wg_c, wg_r):
    """
    Random workgroup size callback.

    Pick a random workgroup size from the parameters table which is
    smaller than or equal to the max workgroup size.
    """
    wgsize = db.rand_wgsize(max_wgsize - 1)
    scenario = instance.get_string_value(0)

    try:
        db.runtime(scenario, hash_params(*wgsize))
        return wgsize
    except lab.db.Error:
        io.warn("Random lookup failed for", wg_c, wg_r)
        return random_fn(db, instance, max_wgsize, wg_c, wg_r)


def reshape_fn(db, instance, max_wgsize, wg_c, wg_r):
    """
    Reshape callback.

    Iteratively reduce the given wgsize in each dimension until it
    fits within the maximum.

    Raises:

        ReshapeError: If the miminum workgroup size is > max_wgsize.
    """
    # Get the lists of all possible wg_c and wg_r values.
    all_wg_c = db.wg_c
    all_wg_r = db.wg_r

    # Convert predicted wg_c, wg_r values into list indices.
    c = all_wg_c.index(wg_c)
    r = all_wg_r.index(wg_r)

    # Iteratively reduce the workgroup size by walking backwards
    # through the list of all possible values.
    while all_wg_c[c] * all_wg_r[r] >= max_wgsize:
        if c + r == 0:
            raise ReshapeError("Failed to shrink wgsize {0}x{1} <= {2}"
                               .format(all_wg_c[c], all_wg_r[r], max_wgsize))
        # Reduce list indices.
        if c > 1: c -= 1
        if r > 1: r -= 1

    return all_wg_c[c], all_wg_r[r]


def eval_instance(classifier, instance, perf_fn, err_fn):
    # Get relevant values from instance.
    oracle = instance.get_string_value(instance.class_index)
    scenario = instance.get_string_value(0)

    # Classify instance, and convert to params ID.
    value = classifier.classify_instance(instance)
    attr = instance.dataset.attribute(instance.class_index)
    predicted = attr.value(value)


    if predicted == oracle:
        return 1, 0, perf_fn(scenario, predicted, oracle)
    else:
        # Determine if predicted workgroup size is valid or not. A
        # valid predicted is one which is within the max_wgsize for
        # that particular instance.
        max_wgsize_attr = instance.dataset.attribute_by_name("kern_max_wg_size")
        max_wgsize_attr_index = max_wgsize_attr.index
        max_wgsize = int(instance.get_value(max_wgsize_attr_index))
        wg_c, wg_r = unhash_params(predicted)
        is_valid = wg_c * wg_r < max_wgsize

        if is_valid:
            return 0, 0, perf_fn(scenario, predicted, oracle)
        else:
            new_wg_c, new_wg_r = err_fn(instance, max_wgsize, wg_c, wg_r)
            try:
                return 0, 1, perf_fn(scenario, hash_params(new_wg_c, new_wg_r),
                                     oracle)
            except lab.db.Error:
                io.error("Woops!", scenario, max_wgsize, new_wg_c,
                         new_wg_r, err_fn.func.__name__)
                return 0, 1, (0, 0)



def eval_classifier(training, testing, classifier, *args, **kwargs):
    # Create attribute filer.
    rm = WekaFilter(classname="weka.filters.unsupervised.attribute.Remove")
    # Create meta-classifier.
    meta = WekaFilteredClassifier()
    meta.set_property("filter", rm)
    meta.set_property("classifier", classifier)

    # Train classifier.
    meta.build_classifier(training)

    results = [eval_instance(classifier, instance, *args, **kwargs)
               for instance in testing]

    correct, invalid, performance = zip(*results)

    return correct, invalid, performance


def xvalidate_classifier(dataset, nfolds, *args, **kwargs):
    data = [eval_classifier(training, testing, *args, **kwargs)
            for training,testing in dataset.folds(nfolds)]

    transpose = map(list, zip(*data))

    return [lab.flatten(column) for column in transpose]


def summarise_classifier_results(correct, invalid, perfs):
    perfs, speedups = zip(*perfs)

    num_tests = len(correct)

    accuracy = (sum(correct) / num_tests) * 100
    ratio_invalid = (sum(invalid) / num_tests) * 100

    min_perf = min(perfs) * 100
    mean_perf = labmath.geomean(perfs) * 100
    max_perf = max(perfs) * 100

    min_speedup = min(speedups)
    mean_speedup = labmath.geomean(speedups)
    max_speedup = max(speedups)

    return (accuracy, ratio_invalid,
            min_perf, mean_perf, max_perf,
            min_speedup, mean_speedup, max_speedup)


def xvalidate_classifiers(classifiers, err_fns, dataset,
                          nfolds=10, perf_cb=lambda *args: 0):
    """
    Cross validate a set of classifiers and err_fns.
    """
    def _plt_title(classifier, n=60):
        if len(classifier) > n:
            return classifier[0:n - 4] + " ..."
        else:
            return classifier

    # All permutations of classifiers and err_fns.
    combinations = list(product(classifiers, err_fns))

    # Get all results.
    xval_results = [
        (classifier2str(classifier), err_fn.func.__name__,
         xvalidate_classifier(dataset, nfolds, classifier, perf_cb, err_fn))
        for classifier,err_fn in combinations
    ]

    # Plot all speedups.
    for i in range(0, len(xval_results), 3):
        for j in range(3):
            row = xval_results[i + j]
            _,Speedups = zip(*sorted(row[2][2], key=lambda x: x[1],
                                     reverse=True))
            err_fn = row[1]
            plt.plot(Speedups, "o-", label=err_fn)


        classifier = row[0]
        plot_name = sanitise_classifier_str(classifier)

        io.info("Plotting", plot_name, "...")

        plt.title(_plt_title(classifier))
        plt.ylabel("Speedup over baseline")
        plt.xlabel("Test instances")
        plt.legend()
        plt.tight_layout()
        plt.savefig("img/eval/classifiers/{}.png".format(plot_name))
        plt.close()


    # Summarise results and print a table.
    table = [
        (t[0], t[1]) + summarise_classifier_results(*t[2])
        for t in xval_results
    ]

    str_args = {
        "float_format": lambda f: "{:.2f}".format(f)
    }

    print("Results of {} fold cross-validation:".format(nfolds))
    print()
    print(fmt.table(table, str_args, columns=(
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


def main():
    """
    Evaluate dataset and omnitune performance.
    """
    ml.start()

    fs.rm("img/eval")
    fs.mkdir("img/eval/classifiers")

    nfolds = 10

    # Get the latest dataset from the oracle.
    db = _db.Database(experiment.ORACLE_PATH)
    db.dump_csvs("/tmp/omnitune/csv")
    dataset = Dataset.load("/tmp/omnitune/csv/oracle_params.csv", db)

    one_r = db.one_r()
    print("ONE R:", one_r[0])
    summarise_perfs(db.perf_param(one_r[0]).values())

    classifiers = (
        ml.ZeroR(),
        ml.SMO(),
        ml.SimpleLogistic(),
        ml.RandomForest(),
        ml.NaiveBayes(),
        ml.J48(),
    )

    err_fns = (
        partial(default_fn, unhash_params(one_r[0])),
        partial(random_fn, db),
        partial(reshape_fn, db)
    )

    # Performance function.
    perf_cb = partial(perf_fn, db, one_r[0])

    xvalidate_classifiers(classifiers, err_fns, dataset,
                          nfolds=nfolds, perf_cb=perf_cb)

    ml.stop()


if __name__ == "__main__":
    main()
