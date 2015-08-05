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
from omnitune.skelcl import space
from omnitune.skelcl.migrate import migrate
from omnitune.skelcl.dataset import Dataset, RegressionDataset

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
def reshape(db, scenario, max_wgsize, wg_c, wg_r):
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
    return random.choice(db.W_legal(scenario))


def reshape_fn(db, instance, max_wgsize, wg_c, wg_r, baseline):
    """
    Reshape callback.

    Iteratively reduce the given wgsize in each dimension until it
    fits within the maximum.
    """
    scenario = instance.get_string_value(0)

    return reshape(db, scenario, max_wgsize, wg_c, wg_r)


def eval_classifier_instance(job, db, classifier, instance,
                             err_fn, training):
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

    correct = 1 if predicted == oracle else 0
    if correct:
        illegal = 0
        refused = 0
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

        illegal = 0 if wg_c * wg_r < max_wgsize else 1
        if illegal:
            refused = 0
        else:
            try:
                db.runtime(scenario, predicted)
                refused = 0
            except lab.db.Error:
                refused = 1

        if illegal or refused:
            predicted = err_fn(instance, max_wgsize, wg_c, wg_r, baseline)

        performance, speedup = perf_fn(db, scenario, predicted,
                                       oracle, baseline)

    db.add_classification_result(job, classifier,
                                 err_fn, training, scenario,
                                 oracle, predicted, baseline, correct, illegal,
                                 refused, performance, speedup)


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

    # # Cross validation using only real data.
    # real_only = Dataset(testing)
    # folds = real_only.folds(nfolds, SEED)
    # print()
    # io.info("CROSS VALIDATION: REAL ONLY")
    # io.info("Size of training set:", folds[0][0].num_instances)
    # io.info("Size of testing set: ", folds[0][1].num_instances)
    # for i,fold in enumerate(folds):
    #     training, testing = fold
    #     io.debug("Cross-validating classifiers, fold", i + 1, "of", nfolds)
    #     eval_fn("xval_real", training, testing)

    # # Leave-one-out validation across architectures.
    # folds = dataset.arch_folds(db)
    # nfolds = len(folds)
    # print()
    # io.info("CROSS-ARCHITECTURE VALIDATION:")
    # for i,fold in enumerate(folds):
    #     training, testing = fold
    #     io.debug("Cross-architecture validating classifiers, fold", i + 1,
    #              "of", nfolds)
    #     eval_fn("arch", training, testing)


def eval_regression(job, db, classifier, instance, dataset, add_cb):
    actual = instance.get_value(instance.class_index)
    scenario = instance.get_string_value(0)

    wg_c = int(instance.get_value(instance.num_attributes - 3))
    wg_r = int(instance.get_value(instance.num_attributes - 2))
    params = hash_params(wg_c, wg_r)

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

        io.debug("   ", job, basename, testing.num_instances, "instances")
        for instance in testing:
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


def get_best_runtime_regression(db, scenario, classifier, job):
    predictions = db.runtime_predictions(scenario, classifier, job)
    try:
        best = min(predictions, key=lambda x: x[1])
        return best[0]
    except ValueError as e:
        print("No runtime predictions for scenario", scenario, "job", job)
        print(e)
        lab.exit(1)

def get_best_speedup_regression(db, scenario, classifier, job):
    predictions = db.speedup_predictions(scenario, classifier, job)
    try:
        best = max(predictions, key=lambda x: x[1])
        return best[0]
    except ValueError as e:
        print("No speedup predictions for scenario", scenario, "job", job)
        print(e)
        lab.exit(1)


def classification(db, nfolds=10):
    dataset = Dataset.load("~/data/msc-thesis/csv/oracle_params.csv", db)

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

    db.empty_table("classification_results")
    eval_fn = partial(eval_classifiers, db, classifiers, err_fns)
    run_eval(db, dataset, eval_fn, "classification")


def regression(db, path, add_cb, get_prediction, add_classification_cb):
    dataset = RegressionDataset.load(path, db)

    db.empty_table("runtime_classification_results")
    job = "xval"
    baseline = "4x4"

    # Convert nominal to binary.
    n2b = WekaFilter(classname="weka.filters.unsupervised.attribute.NominalToBinary",
                     options=["-R", "2-last"])
    n2b.inputformat(dataset.instances)
    dataset.instances = n2b.filter(dataset.instances)

    classifiers = (
        # ml.LinearRegression(),
        ml.RandomForest(),
        # ml.SMOreg(),
        # ml.ZeroR(),
    )

    folds = dataset.folds(10, seed=SEED)
    for i,fold in enumerate(folds):
        training, testing = fold

        maxwgsize_index = testing.attribute_by_name("kern_max_wg_size").index
        wg_c_index = testing.attribute_by_name("wg_c").index
        wg_r_index = testing.attribute_by_name("wg_r").index

        io.debug("Fold", i+1, "of", len(folds))

        for classifier in classifiers:
            meta = Classifier(classifier)
            meta.build_classifier(training)
            basename = ml.classifier_basename(classifier.classname)
            classifier_id = db.classifier_id(classifier)

            scenarios = set([instance.get_string_value(0)
                             for instance in testing])
            instances = zip(scenarios, [
                (instance for instance in testing if
                 instance.get_string_value(0) == scenario).next()
                for scenario in scenarios
            ])

            for scenario,instance in instances:
                maxwgsize = int(instance.get_value(maxwgsize_index))
                wlegal = space.enumerate_wlegal_params(maxwgsize)
                predictions = []

                for params in wlegal:
                    wg_c, wg_r = unhash_params(params)

                    instance.set_value(wg_c_index, wg_c)
                    instance.set_value(wg_r_index, wg_r)

                    predicted = meta.classify_instance(instance)
                    predictions.append((params, predicted))

                # For speedups, we'd invert this search:
                predictions = sorted(predictions, key=lambda x: x[1])

                row = db.execute(
                    "SELECT "
                    "    oracle_param,"
                    "    oracle_runtime,"
                    "    worst_runtime "
                    "FROM scenario_stats "
                    "WHERE scenario=?",
                    (scenario,)).fetchone()
                actual = row[:2]

                predicted_range = predictions[-1][1] - predictions[0][1]
                actual_range = row[2] - row[1]

                io.debug("actual best   ", actual[0],
                         round(actual[1], 1), "ms")

                num_instances = 1
                while True:
                    predicted = predictions.pop(0)

                    io.debug("predicted best", predicted[0],
                             round(predicted[1], 1), "ms")

                    try:
                        actual_runtime_of_predicted = db.runtime(scenario,
                                                                 predicted[0])

                        io.debug("prediction inaccuracy", round(abs(predicted[1] - actual_runtime_of_predicted) / actual_runtime_of_predicted, 2))

                        perf = actual[1] / actual_runtime_of_predicted
                        speedup = db.speedup(scenario, baseline, predicted[0])
                        io.debug("performance", round(perf, 2))

                        db.execute("INSERT INTO runtime_classification_results "
                                   "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                   (job, classifier_id, scenario, actual[0],
                                    actual[1], predicted[0], predicted[1],
                                    actual_range, predicted_range,
                                    num_instances,
                                    1 if predicted[0] == actual[0] else 0,
                                    perf, speedup))
                        break

                    except _db.MissingDataError:
                        num_instances += 1
                        pass

                db.commit()
                print()
                #lab.exit()
            #db.commit()


def runtime_regression(db):
    db.empty_table("runtime_regression_results")
    db.empty_table("classification_runtime_regression_results")
    regression(db, "~/data/msc-thesis/csv/runtime_stats.csv",
               db.add_runtime_regression_result,
               get_best_runtime_regression,
               db.add_runtime_classification_result)


def speedup_regression(db):
    db.empty_table("speedup_regression_results")
    db.empty_table("classifcation_speedup_regression_results")
    regression(db, "~/data/msc-thesis/csv/speedup_stats.csv",
               db.add_speedup_regression_result,
               get_best_speedup_regression,
               db.add_speedup_classification_result)


def eval_linear_models(db, models):
    rows = db.execute(
        "SELECT "
        "    scenario_stats.scenario, "
        "    kernels.max_wg_size, "
        "    scenario_stats.oracle_param "
        "FROM scenarios "
        "LEFT JOIN scenario_stats "
        "  ON scenarios.id=scenario_stats.scenario "
        "LEFT JOIN kernels "
        "  ON scenarios.kernel=kernels.id"
    ).fetchall()

    baseline = db.one_r()[0]

    prof.start("Linear models")
    for scenario,max_wgsize,oracle in rows:
        for model in models:
            wg_c, wg_r = model.predict(scenario, max_wgsize, oracle)

            try:
                prediction = hash_params(wg_c, wg_r)
                illegal = 0 if wg_c * wg_r < max_wgsize else 1
                correct = 1 if prediction == oracle else 0
                db.runtime(scenario, prediction)
                refused = 0

                reshape_param = prediction
                reshape_perf, reshape_speedup = perf_fn(db, scenario,
                                                        prediction, oracle,
                                                        baseline)
                baseline_perf, baseline_speedup = reshape_perf, reshape_speedup
                random_param = prediction
                random_perf, random_speedup = reshape_perf, reshape_speedup
            except lab.db.Error:
                refused = not illegal
                reshape_param = reshape(db, scenario, max_wgsize, wg_c, wg_r)
                reshape_perf, reshape_speedup = perf_fn(db, scenario,
                                                        reshape_param,
                                                        oracle, baseline)

                baseline_perf, baseline_speedup = perf_fn(db, scenario,
                                                          baseline, oracle,
                                                          baseline)

                random_param = random.choice(db.W_legal(scenario))
                random_perf, random_speedup = perf_fn(db, scenario,
                                                      random_param, oracle,
                                                      baseline)

            db.add_model_result(model.id(), "reshape_fn", scenario, oracle,
                                reshape_param, correct, illegal, refused,
                                reshape_perf, reshape_speedup)
            db.add_model_result(model.id(), "default_fn", scenario, oracle,
                                baseline, correct, illegal, refused,
                                baseline_perf, baseline_speedup)
            db.add_model_result(model.id(), "random_fn", scenario, oracle,
                                random_param, correct, illegal, refused,
                                random_perf, random_speedup)
    db.commit()
    prof.stop("Linear models")


class Model(object):
    def predict(self, *args, **kwargs):
        pass
    def id(self, ):
        pass


class LinearModel(Model):
    def __init__(self, c_multiplier, r_multiplier):
        self.c = c_multiplier
        self.r = r_multiplier

    def predict(self, scenario, max_wgsize, *args, **kwargs):
        return max_wgsize * self.c, max_wgsize * self.r

    def id(self):
        return "l-{}-{}".format(self.c, self.r)


def linear_models(db):
    models = [
        LinearModel(0.15, 0.05),
    ]
    db.empty_table("model_results")
    eval_linear_models(db, models)


def main():
    """
    Evaluate dataset and omnitune performance.
    """
    ml.start()
    db = migrate(_db.Database(experiment.ORACLE_PATH))

    # linear_models(db)
    # classification(db)
    runtime_regression(db)
    # speedup_regression(db)

    ml.stop()


if __name__ == "__main__":
    main()
