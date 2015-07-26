#!/usr/bin/env python2
from __future__ import division
from __future__ import print_function

import json
import math
import random
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
from weka.core.dataset import Instances as WekaInstances

import labm8 as lab
from labm8 import fmt
from labm8 import fs
from labm8 import io
from labm8 import math as labmath
from labm8 import ml
from labm8 import prof
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


class Dataset(ml.Dataset):

    def split_synthetic_real(self, db):
        """
        Split dataset based on whether scenario is a synthetic kernel or
        not.

        Returns:

           (WekaInstances, WekaInstances): Only instances from
             synthetic and real benchmarks, respectively.
        """
        real_scenarios = db.real_scenarios

        synthetic = self.copy(self.instances)
        real = self.copy(self.instances)

        # Loop over all instances from last to first.
        for i in range(self.instances.num_instances - 1, -1, -1):
            instance = self.instances.get_instance(i)
            scenario = instance.get_string_value(0)
            if scenario in real_scenarios:
                real.delete(i)
            else:
                synthetic.delete(i)

        return synthetic, real

    def arch_folds(self, db):
        """
        Split dataset to a list of leave-one-out instances, one for each
        architecture.

        Returns:

           list of (WekaInstances, WekaInstances) tuples: A list of
             training, testing pairs, where the training instances
             exclude all scenarios from a specific architecture, and
             the testing instances include only that architecture..
        """
        folds = []

        for device in db.devices:
            device_scenarios = db.scenarios_for_device(device)
            testing = self.copy(self.instances)
            training = self.copy(self.instances)

            # Loop over all instances from last to first.
            for i in range(self.instances.num_instances - 1, -1, -1):
                instance = self.instances.get_instance(i)
                scenario = instance.get_string_value(0)
                if scenario in device_scenarios:
                    training.delete(i)
                else:
                    testing.delete(i)

            folds.append((training, testing))

        return folds

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

        # Load data from CSV.
        dataset = Dataset.load_csv(path, options=force_nominal)
        dataset.__class__ = Dataset

        # Set class index and database connection.
        dataset.class_index = -1
        dataset.db = db

        # Create string->nominal type attribute filter, ignoring the first
        # attribute (scenario ID), since we're not classifying with it.
        string_to_nominal = WekaFilter(classname=("weka.filters.unsupervised."
                                                  "attribute.StringToNominal"),
                                       options=["-R", "2-last"])
        string_to_nominal.inputformat(dataset.instances)

        # Create filtered dataset, and swap data around.
        filtered = string_to_nominal.filter(dataset.instances)
        dataset.instances = filtered

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
    W_safe = db.W_safe
    escaped_params = ",".join(['"' + param + '"' for param in W_safe])

    scenarios = [instance.get_string_value(0) for instance in instances]
    escaped_scenarios = ",".join(['"' + scenario + '"'
                                  for scenario in scenarios])

    baseline = db.execute(
        "SELECT runtime_stats.params\n"
        "FROM runtime_stats\n"
        "LEFT JOIN oracle_params AS oracle\n"
        "ON runtime_stats.scenario=oracle.scenario\n"
        "WHERE runtime_stats.params IN ({params})\n"
        "    AND runtime_stats.scenario IN ({scenarios})\n"
        "GROUP BY runtime_stats.params\n"
        "ORDER BY GEOMEAN(oracle.runtime / runtime_stats.mean) DESC\n"
        "LIMIT 1"
        .format(scenarios=escaped_scenarios, params=escaped_params)
    ).fetchone()[0]

    return baseline


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


def random_fn(db, instance, max_wgsize, wg_c, wg_r, baseline):
    """
    Random workgroup size callback.

    Pick a random workgroup size from the parameters table which is
    smaller than or equal to the max workgroup size.
    """
    scenario = instance.get_string_value(0)
    W_legal = db.W_legal(scenario)
    return random.choice(W_legal)


def reshape_fn(db, instance, max_wgsize, wg_c, wg_r, baseline):
    """
    Reshape callback.

    Iteratively reduce the given wgsize in each dimension until it
    fits within the maximum.
    """
    scenario = instance.get_string_value(0)
    W_legal = db.W_legal(scenario)

    min_distance = float("inf")
    min_param = None

    # Find the *legal* parameter which is closest to the predicted by
    # calculating the distance to each and returning the smallest.
    for param in W_legal:
        p_c, p_r = unhash_params(param)
        distance = math.sqrt((wg_c - p_c) ** 2 + (wg_r - p_r) ** 2)
        if distance < min_distance:
            min_distance = distance
            min_param = param

    return min_param


def eval_classifier_instance(job, db, classifier, instance,
                             err_fn, training):
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

        try:
            db.runtime(scenario, predicted)
            is_valid = True
        except lab.db.Error:
            is_valid = False

        if is_valid:
            correct = 0
            invalid = 0
            performance, speedup = perf_fn(db, scenario, predicted,
                                           oracle, baseline)
        else:
            wg_c, wg_r = unhash_params(predicted)
            correct = 0
            invalid = 1
            predicted = err_fn(instance, max_wgsize, wg_c, wg_r, baseline)
            performance, speedup = perf_fn(db, scenario, predicted,
                                           oracle, baseline)

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


def eval_classifiers(db, classifiers, err_fns, job, training, testing):
    """
    Cross validate a set of classifiers and err_fns.
    """
    for classifier in classifiers:
        meta = Classifier(classifier)
        meta.build_classifier(training)
        basename = ml.classifier_basename(classifier.classname)

        for err_fn in err_fns:
            io.debug(job, basename, err_fn.func.__name__, testing.num_instances)
            for j,instance in enumerate(testing):
                eval_classifier_instance(job, db, meta, instance, err_fn,
                                         training)
            db.commit()


def run_eval(db, dataset, eval_fn, eval_type="", nfolds=10):
    # Cross validation using both synthetic and real data.
    folds = dataset.folds(nfolds, seed=SEED)
    print()
    io.info("CROSS VALIDATION")
    io.info("Size of training set:", folds[0][0].num_instances)
    io.info("Size of testing set: ", folds[0][1].num_instances)

    for i,fold in enumerate(folds):
        training, testing = fold
        io.debug("Cross-validating", eval_type, "- fold", i + 1, "of", nfolds)
        eval_fn("xval", training, testing)

    # Training on synthetic data, testing on real data.
    training, testing = dataset.split_synthetic_real(db)
    print()
    io.info("VALIDATION: REAL ONLY")
    io.info("Size of training set:", training.num_instances)
    io.info("Size of testing set: ", testing.num_instances)
    eval_fn("synthetic_real", training, testing)

    # Cross validation using only real data.
    real_only = Dataset(testing)
    folds = real_only.folds(nfolds, SEED)
    print()
    io.info("CROSS VALIDATION: REAL ONLY")
    io.info("Size of training set:", folds[0][0].num_instances)
    io.info("Size of testing set: ", folds[0][1].num_instances)
    for i,fold in enumerate(folds):
        training, testing = fold
        io.debug("Cross-validating classifiers, fold", i + 1, "of", nfolds)
        eval_fn("xval_real", training, testing)

    # Leave-one-out validation across architectures.
    folds = dataset.arch_folds(db)
    nfolds = len(folds)
    print()
    io.info("CROSS-ARCHITECTURE VALIDATION:")
    for i,fold in enumerate(folds):
        training, testing = fold
        io.debug("Cross-architecture validating classifiers, fold", i + 1,
                 "of", nfolds)
        eval_fn("arch", training, testing)


def eval_regression(job, db, classifier, instance, dataset, add_cb):
    actual = instance.get_value(instance.class_index)
    scenario = instance.get_string_value(0)
    params = instance.get_string_value(102)

    predicted = classifier.classify_instance(instance)
    norm_predicted = predicted / actual
    norm_error = abs(norm_predicted - 1)

    add_cb(job, classifier, dataset, scenario, params, actual, predicted,
           norm_predicted, norm_error)


def eval_regressors(db, classifiers, add_cb, job, training, testing):
    for classifier in classifiers:
        meta = Classifier(classifier)
        meta.build_classifier(training)
        basename = ml.classifier_basename(classifier.classname)

        for j,instance in enumerate(testing):
            io.debug(job, basename, j + 1, "of", testing.num_instances)
            eval_regression(job, db, meta, instance, training, add_cb)
        db.commit()


def eval_regressor_classification_instance(job, db, classifier, scenario,
                                           get_prediction, add_cb, training):
    """
    Returns:

       (int, float, float): From first to last: Correct, Performance
         relative to oracle, Speedup over one_r.
    """
    oracle = db.oracle_param(scenario)

    # Get default value.
    try:
        baseline = training.default
    except AttributeError:
        training.default = get_one_r(db, training)
        baseline = training.default

    # Classify instance.
    predicted = get_prediction(db, scenario, classifier, job)

    correct = 1 if predicted == oracle else 0
    performance, speedup = perf_fn(db, scenario, predicted,
                                   oracle, baseline)

    # Add result to database.
    add_cb(job, classifier, scenario, oracle, predicted, baseline,
           correct, performance, speedup)


def eval_regressor_classifiers(db, classifiers, get_prediction, add_cb,
                               job, training, testing):
    # Get all *unique* scenarios from training set. This is because
    # training sets may have multiple entries for scenarios, one for
    # each param.
    scenarios = set([instance.get_string_value(0) for instance in testing])

    for classifier in classifiers:
        basename = ml.classifier_basename(classifier.classname)

        for j,scenario in enumerate(scenarios):
            io.debug(job, basename, j + 1, "of", len(scenarios))
            eval_regressor_classification_instance(
                job, db, classifier, scenario, get_prediction, add_cb, training
            )
        db.commit()


def get_best_runtime_regression(db, scenario, job):
    predictions = db.runtime_predictions(scenario, job)
    try:
        best = min(predictions, key=lambda x: x[1])
        return best[0]
    except ValueError as e:
        print("No runtime predictions for scenario", scenario, "job", job)
        print(e)
        lab.exit(1)


def get_best_speedup_regression(db, scenario, job):
    predictions = db.speedup_predictions(scenario, job)
    try:
        best = max(predictions, key=lambda x: x[1])
        return best[0]
    except ValueError as e:
        print("No speedup predictions for scenario", scenario, "job", job)
        print(e)
        lab.exit(1)


def classification(db, nfolds=10):
    dataset = Dataset.load("/tmp/omnitune/csv/oracle_params.csv", db)

    classifiers = (
        ml.J48(),
        ml.NaiveBayes(),
        ml.RandomForest(),
        ml.SimpleLogistic(),
        ml.SMO(),
        ml.ZeroR(),
    )

    err_fns = (
        partial(default_fn, db),
        partial(random_fn, db),
        partial(reshape_fn, db),
    )

    eval_fn = partial(eval_classifiers, db, classifiers, err_fns)
    run_eval(db, dataset, eval_fn, "classification")


def regression(db, path, add_cb, get_prediction, add_classification_cb):
    dataset = Dataset.load(path, db)

    classifiers = (
        ml.LinearRegression(),
        ml.RandomForest(),
        #ml.SMOreg(),
        ml.ZeroR(),
    )

    eval_fn = partial(eval_regressors, db, classifiers, add_cb)
    run_eval(db, dataset, eval_fn, "regressors")

    eval_fn = partial(eval_regressor_classifiers, db, classifiers,
                      get_prediction, add_classification_cb)
    run_eval(db, dataset, eval_fn, "regressor classifiers")


def runtime_regression(db):
    regression(db, "/tmp/omnitune/csv/runtime_stats.csv",
               db.add_runtime_regression_result,
               get_best_runtime_regression,
               db.add_runtime_classification_result)


def speedup_regression(db):
    regression(db, "/tmp/omnitune/csv/speedup_stats.csv",
               db.add_speedup_regression_result,
               get_best_speedup_regression,
               db.add_speedup_classification_result)


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
        "runtime_regression_results",
        "speedup_regression_results",
        "classification_runtime_regression_results",
        "classification_speedup_regression_results",
    ]
    for table in tables:
        db.empty_table(table)


    classification(db)
    runtime_regression(db)
    speedup_regression(db)

    ml.stop()


if __name__ == "__main__":
    main()
