#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import json
import re

from functools import partial
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
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
from labm8 import text
from labm8.db import where

import omnitune
from omnitune.skelcl import db as _db
from omnitune.skelcl import hash_params
from omnitune.skelcl import unhash_params
from omnitune.skelcl.migrate import migrate

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

CLASSIFIER_NAME_LEN = 25

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


def get_one_r(db, instances):
    """
    Get the OneR result for a set of instances.

    Arguments:

        db (skelcl.Database): Database for performing lookups.
        instances (weka.Instances): Instances to get OneR of.

    Returns:

        (int, int): wg_c and wg_r values of the OneR for the given
          instances.
    """
    # Get the list of scenarios in instances.
    scenarios = [instance.get_string_value(0) for instance in instances]
    escaped_scenarios = ",".join(['"' + scenario + '"'
                                  for scenario in scenarios])

    # Get the list of params for instances.
    params = [row[0] for row in db.execute(
        "SELECT DISTINCT params\n"
        "FROM runtime_stats\n"
        "WHERE scenario IN ({scenarios})".format(scenarios=escaped_scenarios)
    )]

    # Calculate mean performance of each param.
    avgs = [(param, db.perf_param_avg(param)) for param in params]

    # Select the param with the best performance.
    return max(avgs, key=lambda x: x[1])[0]


def perf_fn(db, scenario, predicted, oracle, baseline):
    """
    Performance callback.

    Arguments:

        db (skelcl.Database): Database object.
        scenario (str): Scenario ID.
        predicted (str): Predicted parameters ID.
        oracle (str): Oracle parameters ID.

    Returns:

        (float, float, str): Ratio of performance to the oracle,
          speedup over baseline params, and baseline params ID.
    """
    speedup = db.speedup(scenario, baseline, predicted)

    if predicted == oracle:
        return 1, speedup
    else:
        return db.perf(scenario, predicted), speedup


####################
# err_fn callbacks #
####################
def default_fn(db, instance, max_wgsize, wg_c, wg_r, baseline):
    """
    Default value parameter callback.

    Simply returns the supplied default argument.
    """
    return baseline


def random_fn(db, instance, max_wgsize, wg_c, wg_r, basline):
    """
    Random workgroup size callback.

    Pick a random workgroup size from the parameters table which is
    smaller than or equal to the max workgroup size.
    """
    wgsize = db.rand_wgsize(max_wgsize - 1)
    scenario = instance.get_string_value(0)

    try:
        db.runtime(scenario, hash_params(*wgsize))
        return hash_params(*wgsize)
    except lab.db.Error:
        io.warn("Random lookup failed for", wg_c, wg_r)
        return random_fn(db, instance, max_wgsize, wg_c, wg_r)


def reshape_fn(db, instance, max_wgsize, wg_c, wg_r, baseline):
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

    return hash_params(all_wg_c[c], all_wg_r[r])


def eval_instance(job, db, classifier, instance, perf_fn, err_fn, training):
    """
    Returns:

       (int, int, float, float): From first to last: Correct, Invalid,
         Performance relative to oracle, Speedup over one_r.
    """
    # Get relevant values from instance.
    oracle = instance.get_string_value(instance.class_index)
    scenario = instance.get_string_value(0)

    # Get default value.
    try:
        baseline = training.default
    except AttributeError:
        training.default = get_one_r(db, training)
        baseline = training.default

    # Classify instance, and convert to params ID.
    value = classifier.classify_instance(instance)
    attr = instance.dataset.attribute(instance.class_index)
    predicted = attr.value(value)

    if predicted == oracle:
        correct = 1
        invalid = 0
        performance, speedup = perf_fn(db, scenario, predicted,
                                       oracle, baseline)
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
            correct = 0
            invalid = 0
            performance, speedup = perf_fn(db, scenario, predicted,
                                           oracle, baseline)
        else:
            correct = 0
            invalid = 1
            predicted = err_fn(instance, max_wgsize, wg_c, wg_r, baseline)
            try:
                performance, speedup = perf_fn(db, scenario, predicted,
                                               oracle, baseline)
            except lab.db.Error:
                performance, speedup, baseline = 0, 0, "Null"
                io.error("Woops!", scenario, max_wgsize, predicted,
                         err_fn.func.__name__)

    db.add_classification_result(job, classifier,
                                 err_fn, training, scenario,
                                 oracle, predicted, baseline, correct, invalid,
                                 performance, speedup)


class Classifier(WekaFilteredClassifier):

    def __init__(self, classifier):
        super(Classifier, self).__init__()

        # Create attribute filer.
        rm = WekaFilter(classname="weka.filters.unsupervised.attribute.Remove",
                        options=["-R", "1"])

        self.classifier = classifier

        self.set_property("filter", rm)
        self.set_property("classifier", classifier)

    def __repr__(self):
        return " ".join([self.classifier.classname,] + self.classifier.options)

    def __str__(self):
        return self.__repr__()


def xvalidate_classifiers(job, db, classifiers, err_fns, perf_fn, dataset,
                          nfolds=10):
    """
    Cross validate a set of classifiers and err_fns.
    """
    # Generate training and testing datasets.
    folds = dataset.folds(nfolds)
    io.info("Size of training set:", folds[0][0].num_instances)
    io.info("Size of testing set: ", folds[0][1].num_instances)

    for i,fold in enumerate(folds):
        training, testing = fold
        for classifier in classifiers:
            meta = Classifier(classifier)
            meta.build_classifier(training)

            for err_fn in err_fns:
                io.info("Evaluating fold", i + 1, "with",
                        err_fn.func.__name__,
                        text.truncate(str(classifier), 40))

                for i,instance in enumerate(testing):
                    io.debug(i)
                    eval_instance(job, db, meta, instance,
                                  perf_fn, err_fn, training)
                db.commit()

def classification(db, nfolds=10):
    dataset = Dataset.load("/tmp/omnitune/csv/oracle_params.csv", db)

    classifiers = (
        ml.ZeroR(),
        ml.SMO(),
        ml.SimpleLogistic(),
        ml.RandomForest(),
        ml.NaiveBayes(),
        ml.J48(),
    )

    err_fns = (
        partial(default_fn, db),
        partial(random_fn, db),
        partial(reshape_fn, db),
    )

    xvalidate_classifiers("xval_classifiers", db, classifiers, err_fns,
                          perf_fn, dataset, nfolds=nfolds)


def eval_runtime(job, db, classifier, instance, dataset):
    actual = instance.get_value(instance.class_index)
    scenario = instance.get_string_value(0)
    params = instance.get_string_value(102)

    predicted = classifier.classify_instance(instance)
    norm_predicted = predicted / actual
    norm_error = abs(norm_predicted - 1)

    db.add_runtime_regression_result(job, classifier, dataset, scenario,
                                     actual, predicted, norm_predicted,
                                     norm_error)


def xvalidate_runtimes(job, db, classifiers, dataset, nfolds):
    # Generate training and testing datasets.
    folds = dataset.folds(nfolds)
    io.info("Size of training set:", folds[0][0].num_instances)
    io.info("Size of testing set: ", folds[0][1].num_instances)

    for i,fold in enumerate(folds):
        training, testing = fold
        for classifier in classifiers:
            meta = Classifier(classifier)
            meta.build_classifier(training)

            io.info("Evaluating fold", i + 1, "with",
                    text.truncate(str(classifier), 40))

            for i,instance in enumerate(testing):
                io.debug(i)
                eval_runtime(job, db, meta, instance, training)
            db.commit()


def runtime_regression(db, nfolds=10):
    dataset = Dataset.load("/tmp/omnitune/csv/runtime_stats.csv", db)

    classifiers = (
        ml.ZeroR(),
    )

    xvalidate_runtimes("xval_runtimes", db, classifiers, dataset, nfolds=nfolds)



def main():
    """
    Evaluate dataset and omnitune performance.
    """
    ml.start()

    # Get the latest dataset from the oracle.
    db = migrate(_db.Database(experiment.ORACLE_PATH))
    db.dump_csvs("/tmp/omnitune/csv")

    # Empty old data.
    tables = [
        "classifiers",
        "err_fns",
        "ml_datasets",
        "ml_jobs",
        "classification_results",
        "runtime_regression_results"
    ]
    for table in tables:
        db.empty_table(table)


    classification(db)
    runtime_regression(db)

    ml.stop()


if __name__ == "__main__":
    main()
